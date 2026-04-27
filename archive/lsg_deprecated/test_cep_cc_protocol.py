from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cep_cc.env import ContinuousObjectSelectionEnv, EnvConfig
from experiments.cep_cc.losses import effective_dimension, task_loss
from experiments.cep_cc.models import ModelConfig, SpeakerListenerSystem
from experiments.cep_cc.run_experiment import (
    run_c1f_ablation,
    run_c1g_geometry_audit,
    run_c1h_consistency_partitioning,
    run_c1i_factor_segment_audit,
    run_c1j_compositional_variant,
    run_c1k_temporal_memory_variant,
    run_c1l_temporal_bootstrap,
    run_c2a_semantic_stability,
    run_c2b_cluster_invariant_alignment,
    run_c2c_factor_separated_invariance,
    run_c3a_targeted_segment_intervention,
    run_c3b_targeted_intervention_robustness,
    run_c3c_motion_pressure_variant,
    run_c3d_motion_decoupling_variant,
    run_c3e_motion_readout_audit,
    run_c3f_factorized_pressure_objective,
    run_c1c_calibration,
    run_lambda_sweep,
    run_multiseed_compression_sweep,
    run_train_eval,
    run_two_stage_compression,
)


def test_environment_emits_expected_shapes() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig())
    batch = env.sample_batch(7, seed=0)
    assert batch.speaker_obs.shape == (7, env.speaker_obs_dim)
    assert batch.listener_obs.shape == (7, env.listener_obs_dim)
    assert batch.object_states.shape == (7, 4, 5)
    assert batch.latent_factors.shape == (7, 4, 4)
    assert batch.target_index.shape == (7,)
    assert batch.target_index.dtype == torch.long


def test_compositional_environment_emits_expected_shapes() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig(rule_mode="compositional"))
    batch = env.sample_batch(7, seed=0)
    assert batch.speaker_obs.shape == (7, env.speaker_obs_dim)
    assert batch.listener_obs.shape == (7, env.listener_obs_dim)
    assert batch.latent_factors.shape == (7, 4, 4)
    assert batch.target_index.shape == (7,)


def test_temporal_memory_environment_emits_expected_shapes() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig(rule_mode="temporal_memory"))
    batch = env.sample_batch(7, seed=0)
    assert batch.speaker_obs.shape == (7, env.speaker_obs_dim)
    assert batch.listener_obs.shape == (7, env.listener_obs_dim)
    assert batch.latent_factors.shape == (7, 4, 4)
    assert batch.target_index.shape == (7,)


def test_nuisance_environment_emits_expected_shapes() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig(rule_mode="temporal_memory", nuisance_mode="mirror_x"))
    batch = env.sample_batch(7, seed=0)
    assert batch.speaker_obs.shape == (7, env.speaker_obs_dim)
    assert batch.listener_obs.shape == (7, env.listener_obs_dim)
    assert batch.latent_factors.shape == (7, 4, 4)
    assert batch.target_index.shape == (7,)


def test_motion_pressure_environment_emits_expected_shapes() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig(rule_mode="motion_pressure"))
    batch = env.sample_batch(7, seed=0)
    assert batch.speaker_obs.shape == (7, env.speaker_obs_dim)
    assert batch.listener_obs.shape == (7, env.listener_obs_dim)
    assert batch.latent_factors.shape == (7, 4, 4)
    assert batch.target_index.shape == (7,)


def test_motion_decoupled_environment_emits_expected_shapes() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig(rule_mode="motion_decoupled"))
    batch = env.sample_batch(7, seed=0)
    assert batch.speaker_obs.shape == (7, env.speaker_obs_dim)
    assert batch.listener_obs.shape == (7, env.listener_obs_dim)
    assert batch.latent_factors.shape == (7, 4, 4)
    assert batch.target_index.shape == (7,)


def test_motion_decoupled_multi_factor_values_are_supported() -> None:
    from experiments.cep_cc.run_experiment import _target_factor_values

    env = ContinuousObjectSelectionEnv(EnvConfig(rule_mode="motion_decoupled"))
    batch = env.sample_batch(7, seed=0)
    values = _target_factor_values(batch.latent_factors, batch.object_states, batch.target_index, "q+motion+relation")
    assert values.shape == (7, 3)
    assert values.dtype.is_floating_point


def test_forward_pass_uses_continuous_communication() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig())
    config = ModelConfig(env.speaker_obs_dim, env.listener_obs_dim)
    system = SpeakerListenerSystem(config, seed=0)
    batch = env.sample_batch(5, seed=1)
    out = system(batch.speaker_obs, batch.listener_obs)
    assert out["logits"].shape == (5, 4)
    assert out["comm"].shape == (5, config.comm_steps, config.comm_dim)
    assert out["comm"].dtype.is_floating_point
    assert torch.unique(out["comm"]).numel() > config.comm_dim


def test_effective_dimension_is_positive() -> None:
    values = torch.randn(16, 8, 6)
    dim = effective_dimension(values)
    assert float(dim) > 0.0


def test_tiny_train_eval_runs() -> None:
    metrics = run_train_eval(episodes=4, batch_size=16, seed=2, lambda_comm=0.001)
    assert "task_accuracy" in metrics
    assert "comm_effective_dim" in metrics
    assert "prototype_reuse_rate" in metrics
    assert 0.0 <= metrics["task_accuracy"] <= 1.0
    assert metrics["comm_effective_dim"] > 0.0


def test_lambda_sweep_includes_no_communication_baseline() -> None:
    results = run_lambda_sweep([0.0, 0.001], episodes=3, batch_size=12, seed=3)
    assert "no_communication" in results
    assert "lambda_0" in results
    assert "lambda_0.001" in results
    for metrics in results.values():
        assert "task_accuracy" in metrics
        assert "comm_energy" in metrics


def test_teacher_signal_positive_control_runs() -> None:
    metrics = run_train_eval(episodes=4, batch_size=16, seed=5, baseline="teacher-signal")
    assert "task_accuracy" in metrics
    assert "train_task_accuracy" in metrics
    assert metrics["comm_energy"] > 0.0
    assert metrics["comm_effective_dim"] > 0.0


def test_c1c_calibration_runs_all_conditions() -> None:
    results = run_c1c_calibration(episodes=2, batch_size=8, seed=6)
    assert "no_communication" in results
    assert "teacher_signal" in results
    assert "learned_high_bandwidth" in results
    assert "learned_compressed_0.001" in results
    assert "learned_distilled" in results
    for metrics in results.values():
        assert "task_accuracy" in metrics
        assert "train_task_accuracy" in metrics


def test_multiseed_compression_sweep_runs() -> None:
    results = run_multiseed_compression_sweep(
        seeds=[0, 1],
        lambdas=[0.0, 0.001],
        episodes=2,
        batch_size=8,
        lr=0.002,
    )
    assert "seed_0:no_communication" in results
    assert "seed_0:lambda_0" in results
    assert "seed_1:lambda_0.001" in results
    for metrics in results.values():
        assert "task_accuracy" in metrics
        assert "comm_effective_dim" in metrics


def test_two_stage_compression_runs() -> None:
    results = run_two_stage_compression(
        seeds=[0],
        lambdas=[0.001],
        stage1_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
    )
    assert "seed_0:stage1_high_bandwidth" in results
    assert "seed_0:stage2_lambda_0.001" in results
    tuned = results["seed_0:stage2_lambda_0.001"]
    assert "delta_task_accuracy" in tuned
    assert "stage1_task_accuracy" in tuned


def test_c1f_ablation_runs() -> None:
    results = run_c1f_ablation(
        seeds=[0],
        lambdas=[0.03],
        stage1_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
    )
    assert "joint:seed_0:stage2_lambda_0.03" in results
    assert "freeze_listener:seed_0:stage2_lambda_0.03" in results
    assert "delta_comm_energy" in results["freeze_listener:seed_0:stage2_lambda_0.03"]


def test_c1g_geometry_audit_runs() -> None:
    results = run_c1g_geometry_audit(
        seeds=[0],
        stage1_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        audit_batches=2,
    )
    assert "seed_0:stage1_high_bandwidth" in results
    assert "seed_0:compressed_freeze_listener_lambda_0.1" in results
    compressed = results["seed_0:compressed_freeze_listener_lambda_0.1"]
    assert "audit_target_purity" in compressed
    assert "audit_latent_probe_r2" in compressed


def test_c1h_consistency_partitioning_runs() -> None:
    results = run_c1h_consistency_partitioning(
        seeds=[0],
        consistency_values=[0.01],
        stage1_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        audit_batches=2,
    )
    assert "seed_0:consistency_0.01" in results
    row = results["seed_0:consistency_0.01"]
    assert "train_comm_consistency" in row
    assert row["consistency_lambda"] == 0.01


def test_c1i_factor_segment_audit_runs() -> None:
    results = run_c1i_factor_segment_audit(
        seeds=[0],
        factor_names=["q"],
        factor_consistency_values=[0.01],
        stage1_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        audit_batches=2,
    )
    assert "seed_0:factor_q_0.01" in results
    row = results["seed_0:factor_q_0.01"]
    assert "train_factor_consistency" in row
    assert "segment_q_specialization_gap" in row


def test_c1j_compositional_variant_runs() -> None:
    results = run_c1j_compositional_variant(
        seeds=[0],
        factor_names=["q"],
        factor_consistency_values=[0.01],
        stage1_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        audit_batches=2,
    )
    assert "seed_0:no_communication" in results
    assert "seed_0:c1j_factor_q_0.01" in results
    row = results["seed_0:c1j_factor_q_0.01"]
    assert "segment_early_swap_action_change_rate" in row
    assert row["rule_mode_compositional"] == 1.0


def test_c1k_temporal_memory_variant_runs() -> None:
    results = run_c1k_temporal_memory_variant(
        seeds=[0],
        factor_names=["q"],
        factor_consistency_values=[0.01],
        stage1_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        audit_batches=2,
    )
    assert "seed_0:no_communication" in results
    assert "seed_0:c1k_factor_q_0.01" in results
    row = results["seed_0:c1k_factor_q_0.01"]
    assert "segment_early_ablation_action_change_rate" in row
    assert row["rule_mode_temporal_memory"] == 1.0


def test_c1l_temporal_bootstrap_runs() -> None:
    results = run_c1l_temporal_bootstrap(
        seeds=[0],
        bootstrap_modes=["segment_dropout"],
        factor_names=["q"],
        factor_consistency_values=[0.01],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        segment_dropout_prob=0.25,
        audit_batches=2,
    )
    assert "seed_0:segment_dropout_stage1_temporal_memory" in results
    assert "seed_0:segment_dropout_c1l_factor_q_0.01" in results
    row = results["seed_0:segment_dropout_c1l_factor_q_0.01"]
    assert "train_segment_dropout_prob" in row
    assert row["bootstrap_segment_dropout"] == 1.0


def test_c2a_semantic_stability_runs() -> None:
    results = run_c2a_semantic_stability(
        seeds=[0],
        nuisance_modes=["none", "mirror_x"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="q",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c2a_none" in results
    assert "seed_0:c2a_mirror_x" in results
    assert "delta_clean_task_accuracy" in results["seed_0:c2a_mirror_x"]
    assert results["seed_0:c2a_mirror_x"]["nuisance_is_mirror_x"] == 1.0


def test_c2b_cluster_invariant_alignment_runs() -> None:
    results = run_c2b_cluster_invariant_alignment(
        seeds=[0],
        nuisance_modes=["mirror_x"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="q",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c2b_clean" in results
    assert "seed_0:c2b_mirror_x" in results
    row = results["seed_0:c2b_mirror_x"]
    assert "paired_proto_assignment_stability" in row
    assert "paired_hidden_q_corr" in row


def test_c2c_factor_separated_invariance_runs() -> None:
    results = run_c2c_factor_separated_invariance(
        seeds=[0],
        nuisance_modes=["mirror_x"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="q",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c2c_mirror_x" in results
    row = results["seed_0:c2c_mirror_x"]
    assert "paired_q_score_factor_corr" in row
    assert "paired_equivariant_factor_corr" in row


def test_c3a_targeted_segment_intervention_runs() -> None:
    results = run_c3a_targeted_segment_intervention(
        seeds=[0],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="q",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c3a_targeted_intervention" in results
    row = results["seed_0:c3a_targeted_intervention"]
    assert "intervention_best_targeted_ratio" in row
    assert "intervention_q_best_ratio" in row


def test_c3b_targeted_intervention_robustness_runs() -> None:
    results = run_c3b_targeted_intervention_robustness(
        seeds=[0],
        nuisance_modes=["none", "mirror_x"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="q",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c3b_none" in results
    assert "seed_0:c3b_mirror_x" in results
    assert "intervention_relation_best_ratio" in results["seed_0:c3b_mirror_x"]
    assert results["seed_0:c3b_mirror_x"]["nuisance_is_mirror_x"] == 1.0


def test_c3c_motion_pressure_variant_runs() -> None:
    results = run_c3c_motion_pressure_variant(
        seeds=[0],
        nuisance_modes=["none"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="motion",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c3c_none" in results
    row = results["seed_0:c3c_none"]
    assert row["rule_mode_motion_pressure"] == 1.0
    assert "intervention_motion_best_ratio" in row


def test_c3d_motion_decoupling_variant_runs() -> None:
    results = run_c3d_motion_decoupling_variant(
        seeds=[0],
        nuisance_modes=["none"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="motion",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c3d_none" in results
    row = results["seed_0:c3d_none"]
    assert row["rule_mode_motion_decoupled"] == 1.0
    assert "intervention_motion_best_ratio" in row


def test_c3e_motion_readout_audit_runs() -> None:
    results = run_c3e_motion_readout_audit(
        seeds=[0],
        nuisance_modes=["none"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_name="motion",
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c3e_none" in results
    row = results["seed_0:c3e_none"]
    assert row["rule_mode_motion_decoupled"] == 1.0
    assert "audit_motion_probe_r2" in row
    assert "segment_late_motion_probe_r2" in row


def test_c3f_factorized_pressure_objective_runs() -> None:
    results = run_c3f_factorized_pressure_objective(
        seeds=[0],
        factor_sets=["motion", "q+motion"],
        stage1_episodes=2,
        curriculum_episodes=2,
        stage2_episodes=2,
        batch_size=8,
        lr=0.002,
        stage2_lr=0.001,
        lambda_comm=0.10,
        factor_consistency=0.01,
        audit_batches=2,
    )
    assert "seed_0:c3f_motion" in results
    assert "seed_0:c3f_q+motion" in results
    row = results["seed_0:c3f_q+motion"]
    assert row["factor_set_has_q"] == 1.0
    assert row["factor_set_has_motion"] == 1.0
    assert row["factor_set_size"] == 2.0
    assert "audit_motion_probe_r2" in row


def test_comm_distillation_run_produces_distill_loss() -> None:
    metrics = run_train_eval(
        episodes=3,
        batch_size=8,
        seed=7,
        baseline="learned",
        lambda_comm_distill=1.0,
    )
    assert "train_comm_distill" in metrics
    assert metrics["train_comm_distill"] >= 0.0


def test_short_training_reduces_fixed_batch_task_loss() -> None:
    env = ContinuousObjectSelectionEnv(EnvConfig())
    config = ModelConfig(env.speaker_obs_dim, env.listener_obs_dim)
    system = SpeakerListenerSystem(config, seed=4)
    optimizer = torch.optim.Adam(system.parameters(), lr=0.01)
    batch = env.sample_batch(32, seed=4)
    initial = task_loss(system(batch.speaker_obs, batch.listener_obs)["logits"], batch.target_index)
    for _ in range(12):
        out = system(batch.speaker_obs, batch.listener_obs)
        loss = task_loss(out["logits"], batch.target_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = task_loss(system(batch.speaker_obs, batch.listener_obs)["logits"], batch.target_index)
    assert float(final.detach()) < float(initial.detach())
