import { useState } from "react";
import { Card, Pill } from "./ui";

// keep in step with docs/TRAINING.md, training/src/drone_model (brain.py, scripted/, policies/), and training/RESULTS.md
const AS_OF = "2026-10-28";

interface Stage {
  id: string;
  title: string;
  where: string;
  kind: "mod" | "learned" | "scripted";
  summary: string;
  detail: string[];
}

const LOOP: Stage[] = [
  {
    id: "camera",
    title: "Camera and state",
    where: "mod",
    kind: "mod",
    summary: "The mod renders what the drone sees",
    detail: [
      "Each step the mod renders the drone's camera and sends the image, depth, and the drone's own state: position, heading, velocity, battery, inventory, range sensors, and its job.",
      "Training arenas also have ground-truth block and entity labels. Those only train the block reader and never reach the drone in flight.",
    ],
  },
  {
    id: "reader",
    title: "Block reader",
    where: "perception/reader.py, checkpoints/reader.pt",
    kind: "learned",
    summary: "Labels every pixel",
    detail: [
      "A segmentation network, a ConvNeXt-Tiny encoder pretrained on ImageNet with a U-Net decoder, that labels each 320x240 frame's pixels as one of 1,303 classes: every block a camera can show and every kind of mob.",
      "Depth turns each labeled pixel into a 3D point. It trains on staged scenes of each block and mob shot from random angles, blocks rendered offline on the GPU from the game's models and textures and mobs and chest-like blocks captured in the game, and the shipped checkpoint was also fine-tuned on game frames of wheat fields. It reaches mean IoU 0.396 on held-out game frames.",
    ],
  },
  {
    id: "memory",
    title: "Memory",
    where: "perception/core.py, memory.py, worldmap.py",
    kind: "scripted",
    summary: "Remembers blocks",
    detail: [
      "Each drone has one perception core that it keeps across its jobs: the reader, the voxel memory, and the world map. The memory and map keep which blocks are where, so the drone can act on things it isn't looking at. A block another read outvotes stops counting as a goal. The memory panel on the Live tab shows it.",
    ],
  },
  {
    id: "planner",
    title: "Planner",
    where: "scripted/jobs.py, goto.py",
    kind: "scripted",
    summary: "Decides what to do next",
    detail: [
      "Hand-written, one per kind of job (the table below). It works out what's surveyed, what's left, and which target to commit to, and names one goal each step: look at a point from a viewpoint, break or place a block, or go to a point. Every break and place is worked from a fixed pose square-on to one face of the block, 2.5 blocks straight out and level for a side face and 2.5 overhead for a top face.",
      "It gets only what the drone perceives, never the true arena layout or the source's blocks.",
    ],
  },
  {
    id: "pilot",
    title: "Pilot",
    where: "policies/skill.py, goto.py",
    kind: "learned",
    summary: "Flies the goal",
    detail: [
      "A learned policy turns the goal plus the camera image and depth into stick input (forward, sideways, up, yaw, pitch) and whether to fire the tool. The cell skill flies the view, break, and place goals of copies and builds. The goto policy flies goto goals and the flights home to charge.",
      "The scripted controller is a dev toggle: python -m drone_model.brain --scripted flies the same goals with hand-written steering, for comparison.",
    ],
  },
  {
    id: "apply",
    title: "Apply and score",
    where: "mod",
    kind: "mod",
    summary: "Moves the drone, scores the step",
    detail: [
      "The mod eases velocity and turning so the camera turns smoothly, moves the drone, and uses the tool.",
      "It scores the step (progress, collisions, turn reversals, the geofence), checks whether the job is done, and sends the next frame.",
    ],
  },
];

const TRAINING: Stage[] = [
  {
    id: "arenas",
    title: "Arenas",
    where: "mod, TrainingArenas.java",
    kind: "mod",
    summary: "Seeded practice worlds",
    detail: [
      "Each seed builds an arena on flat, rough, or cave terrain with obstacles, a structure to copy or a schematic to build, and for gather arenas a stone deposit holding the materials to dig.",
    ],
  },
  {
    id: "demos",
    title: "Planner episodes",
    where: "framework/collect.py",
    kind: "scripted",
    summary: "The planner flies, every step is labeled",
    detail: [
      "The planner and scripted controller fly with deliberate flight noise so they also end up off course. Each step is saved with the controller's clean action as the label.",
    ],
  },
  {
    id: "bc",
    title: "Behavior cloning",
    where: "framework/train.py",
    kind: "learned",
    summary: "The network copies the labels",
    detail: [
      "The policy learns to predict the label from the image, depth, its motion, and the goal. A smoothness term raises the loss whenever the predicted turn flips direction from one step to the next.",
    ],
  },
  {
    id: "dagger",
    title: "DAgger rounds",
    where: "scripts/dagger_skill.ps1, dagger_goto.ps1",
    kind: "learned",
    summary: "The policy flies, the planner corrects",
    detail: [
      "The trained policy flies half the steps while the planner labels every step with what it would do, then the policy retrains on everything. That teaches recovery from its own mistakes, which a plain copy never sees.",
    ],
  },
  {
    id: "evaluate",
    title: "Evaluate",
    where: "framework/evaluate.py, brain.py --task",
    kind: "learned",
    summary: "Fixed seeds, scored",
    detail: [
      "Success rate, turn reversals per 100 steps, and how failed episodes ended, for the policy and for the planner alone.",
      "The policy learns from the planner, so it can at best match the planner. A planner fix comes first, then a DAgger round carries it over.",
    ],
  },
];

const KIND: Record<
  Stage["kind"],
  { label: string; tone: "accent" | "neutral" | "amber" }
> = {
  mod: { label: "mod", tone: "neutral" },
  learned: { label: "learned", tone: "accent" },
  scripted: { label: "scripted", tone: "amber" },
};

const POLICIES = [
  {
    name: "Cell skill",
    file: "checkpoints/skill.pt",
    flies: "View, break, and place goals of copies and builds",
    status:
      "One DAgger round on the square-on work pose: replicate 4 of 5, copy flat 3 of 5, copy rough 0 of 5. Trails the planner, more rounds next",
  },
  {
    name: "Goto policy",
    file: "checkpoints/goto.pt",
    flies: "Goto goals and the flights home",
    status:
      "Retrained on the goto_point and dock_station data, not evaluated since",
  },
  {
    name: "Block reader",
    file: "checkpoints/reader.pt",
    flies: "Perception for every job",
    status:
      "ConvNeXt-Tiny U-Net at 320x240, 1,303 classes, mean IoU 0.396 on held-out game frames",
  },
];

const PLANNERS = [
  [
    "Copy, build",
    "BuildPlanner",
    "Reads the source with the camera or loads the schematic, digs the missing blocks from a materials region, clears what doesn't belong, and places the plan bottom up",
  ],
  [
    "Go to a point",
    "GotoPlanner",
    "Names the arena's point as a goto goal, for training and evaluating the goto policy",
  ],
  [
    "Return home",
    "DockPlanner",
    "Names the spot over the charging station as a goto goal, which the goto policy flies",
  ],
];

function Flow({ stages, loop }: { stages: Stage[]; loop?: boolean }) {
  const [selected, setSelected] = useState(stages[0].id);
  const stage = stages.find((s) => s.id === selected) ?? stages[0];
  return (
    <div className="flex flex-col gap-4">
      <ol className="flex flex-col gap-2 lg:flex-row lg:items-stretch">
        {stages.map((s, i) => (
          <li
            key={s.id}
            className="flex flex-col items-center gap-2 lg:flex-1 lg:flex-row"
          >
            <button
              onClick={() => setSelected(s.id)}
              aria-pressed={s.id === selected}
              className={`flex w-full flex-col gap-1 rounded-md border px-3 py-2 text-left transition-colors lg:h-full ${
                s.id === selected
                  ? "border-accent bg-accent-light"
                  : "border-border bg-card hover:bg-sunken"
              }`}
            >
              <span className="flex items-center justify-between gap-2">
                <span className="text-text-primary text-sm font-semibold">
                  {i + 1}. {s.title}
                </span>
                <Pill tone={KIND[s.kind].tone}>{KIND[s.kind].label}</Pill>
              </span>
              <span className="text-text-muted text-xs">{s.summary}</span>
            </button>
            {i < stages.length - 1 && <Arrow />}
          </li>
        ))}
      </ol>
      {loop && (
        <p className="text-text-dim text-xs">
          After step {stages.length} the next frame starts again at step 1.
        </p>
      )}
      <div className="border-border bg-sunken rounded-md border px-4 py-3">
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-text-primary text-sm font-semibold">
            {stage.title}
          </h3>
          <code className="text-text-muted font-mono text-xs">
            {stage.where}
          </code>
        </div>
        {stage.detail.map((d) => (
          <p key={d} className="text-text-primary mb-1.5 text-sm last:mb-0">
            {d}
          </p>
        ))}
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="text-text-dim h-4 w-4 shrink-0 rotate-90 lg:rotate-0"
      aria-hidden
    >
      <path
        d="M2 8h10M8 4l4 4-4 4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function SystemView() {
  return (
    <div className="flex flex-col gap-4">
      <Card title="How a job runs">
        <p className="text-text-muted mb-4 text-sm">
          The mod runs the drone and the brain (python -m drone_model.brain)
          decides what it does. The planner picks a goal and a learned policy
          flies it. Pick a step for details.
        </p>
        <Flow stages={LOOP} loop />
        <p className="text-text-muted mt-3 text-sm">
          A battery keeper sits beside the planner. Once the charge barely
          covers the trip home and the reserve, it takes over and the goto
          policy flies the drone to its station, then hands back after a full
          charge.
        </p>
      </Card>

      <Card title="How a policy is trained">
        <Flow stages={TRAINING} />
        <p className="text-text-muted mt-3 text-sm">
          The block reader trains on its own from studio frames, rendered
          offline for blocks (collect/render.py) and captured in the game for
          mobs and chest-like blocks (collect/studio.py, train/reader.py).
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="Learned models"
          actions={<span className="text-text-dim text-xs">as of {AS_OF}</span>}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-text-muted text-xs">
                <tr>
                  <th className="py-1 pr-3 font-medium">Model</th>
                  <th className="py-1 pr-3 font-medium">Flies</th>
                  <th className="py-1 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {POLICIES.map((p) => (
                  <tr key={p.name} className="border-border border-t align-top">
                    <td className="py-1.5 pr-3">
                      <div className="font-medium">{p.name}</div>
                      <code className="text-text-dim font-mono text-xs">
                        {p.file}
                      </code>
                    </td>
                    <td className="py-1.5 pr-3">{p.flies}</td>
                    <td className="text-text-muted py-1.5">{p.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Planners by job">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-text-muted text-xs">
                <tr>
                  <th className="py-1 pr-3 font-medium">Job</th>
                  <th className="py-1 pr-3 font-medium">Planner</th>
                  <th className="py-1 font-medium">Does</th>
                </tr>
              </thead>
              <tbody>
                {PLANNERS.map(([job, planner, does]) => (
                  <tr
                    key={planner}
                    className="border-border border-t align-top"
                  >
                    <td className="py-1.5 pr-3">{job}</td>
                    <td className="py-1.5 pr-3 font-mono text-xs">{planner}</td>
                    <td className="text-text-muted py-1.5">{does}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
