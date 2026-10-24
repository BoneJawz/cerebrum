"""Synthetic symbolic acoustic cues for the factor-learning experiment.

Cue IDs and the waveform table are simulator metadata. Processor observations
continue to contain only raw waveform samples and visual arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from pathlib import Path

import numpy as np

from .contracts import Observation
from .environment import LineWorld, MOTION_START_TICK, WorldConfig


@dataclass(frozen=True)
class CueDefinition:
    concept_id: str
    concept_name: str
    waveform_id: str
    category: str
    frequencies_hz: tuple[float, float]
    duration_ticks: int = 2
    nominal_amplitude: float = 0.5


# Nonmonotonic arbitrary tone pairs. Every cue has equal duration and nominal
# amplitude; speed is not mapped to a monotonic carrier or average frequency.
VOCABULARY = (
    CueDefinition("DIR_L", "LEFT", "tone_pair_01", "direction", (320.0, 880.0)),
    CueDefinition("DIR_R", "RIGHT", "tone_pair_02", "direction", (880.0, 480.0)),
    CueDefinition("SPD_1", "SPEED_1", "tone_pair_03", "speed", (440.0, 840.0)),
    CueDefinition("SPD_2", "SPEED_2", "tone_pair_04", "speed", (760.0, 320.0)),
    CueDefinition("SPD_3", "SPEED_3", "tone_pair_05", "speed", (560.0, 680.0)),
)
VOCAB_BY_NAME = {cue.concept_name: cue for cue in VOCABULARY}
VOCAB_BY_ID = {cue.concept_id: cue for cue in VOCABULARY}


def write_vocabulary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("concept_id", "concept_name", "waveform_id",
            "cue_category", "frequencies_hz", "duration_ticks", "nominal_amplitude"))
        writer.writeheader()
        for cue in VOCABULARY:
            writer.writerow({"concept_id":cue.concept_id,"concept_name":cue.concept_name,
                "waveform_id":cue.waveform_id,"cue_category":cue.category,
                "frequencies_hz":";".join(str(v) for v in cue.frequencies_hz),
                "duration_ticks":cue.duration_ticks,"nominal_amplitude":cue.nominal_amplitude})


@dataclass(frozen=True)
class FactorEpisode:
    observations: tuple[Observation, ...]
    direction: int
    speed: int
    initial_position: int
    task: str
    cue_ids: tuple[str, ...]
    cue_segments: tuple[tuple[str, int, int], ...]
    seed: int


class FactorWorld:
    """Existing visual world plus either legacy or sequential symbolic audio."""
    def __init__(self, config: WorldConfig | None = None):
        self.config=config or WorldConfig(pixels=177,ticks=24)
        self.world=LineWorld(self.config)
        self.chunk=self.config.audio_hz//self.config.visual_hz

    @staticmethod
    def schedule(task: str, direction: int, speed: int):
        if task == "combined":
            return (("DIR_L" if direction < 0 else "DIR_R",0,2),
                    ("SILENCE",2,3),
                    (f"SPD_{speed}",3,5))
        if task == "direction_focus":
            return (("DIR_L" if direction < 0 else "DIR_R",0,2),)
        if task == "speed_focus":
            return ((f"SPD_{speed}",0,2),)
        raise ValueError("task must be combined, direction_focus, or speed_focus")

    def _cue_chunk(self, cue: CueDefinition, tick_in_cue: int, seed_rng):
        idx=np.arange(self.chunk,dtype=float)
        amp=cue.nominal_amplitude*float(seed_rng.uniform(.85,1.15))
        phase=float(seed_rng.uniform(0,2*np.pi))
        freq=cue.frequencies_hz[tick_in_cue]
        return amp*np.sin(2*np.pi*freq*idx/self.config.audio_hz+phase)

    def episode(self, seed: int, direction: int, speed: int, initial_position=None,
                task="combined", acoustic_config="B_independent", noise_std=None,
                phase_sign=1):
        if direction not in (-1,1) or speed not in (1,2,3):
            raise ValueError("direction must be +/-1 and speed one of 1, 2, 3")
        rng=np.random.default_rng(seed)
        initial=(self.world.sample_initial_position(rng) if initial_position is None
                 else int(initial_position))
        if acoustic_config not in ("A_legacy","B_independent"):
            raise ValueError("unknown acoustic configuration")
        # Use LineWorld only for the established visual trajectory and its
        # independent episode seeding; the original audio is replaced below.
        phase=float(direction*int(phase_sign)*np.pi/2)
        base=self.world.episode(seed=seed,direction=direction,velocity=speed,
            tone_hz=120+180*speed,initial_position=initial,phase_offset=phase)
        segments=self.schedule(task,direction,speed) if acoustic_config=="B_independent" else ()
        by_tick={}
        for cue_id,start,end in segments:
            by_tick.update({tick:(cue_id,tick-start) for tick in range(start,end)})
        result=[]
        sigma=self.config.noise_std if noise_std is None else float(noise_std)
        for tick,o in enumerate(base):
            if acoustic_config=="A_legacy":
                audio=o.audio
            else:
                if tick in by_tick and by_tick[tick][0]!="SILENCE":
                    cue_id,within=by_tick[tick]
                    audio=self._cue_chunk(VOCAB_BY_ID[cue_id],within,rng)
                else:
                    audio=np.zeros(self.chunk)
                audio=audio+rng.normal(0,sigma,self.chunk)
            result.append(replace(o,audio=np.asarray(audio,dtype=np.float64)))
        cues=tuple(seg[0] for seg in segments if seg[0]!="SILENCE")
        return FactorEpisode(tuple(result),direction,speed,initial,task,cues,
                             tuple(segments),seed)


def target_trajectory(episode: FactorEpisode, input_end_tick: int,
                      horizons=tuple(range(1,13))):
    """Future centers measured relative to the last assimilated observation."""
    current=max(input_end_tick,MOTION_START_TICK)
    return np.asarray([np.sum(episode.observations[current+h].vision*np.arange(
        len(episode.observations[current+h].vision)))/np.sum(episode.observations[current+h].vision)
        for h in horizons],dtype=float)

