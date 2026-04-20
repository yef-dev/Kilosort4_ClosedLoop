import numpy as np
import torch
import pytest

from kilosort import closed_loop, io, template_matching
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.run_kilosort import initialize_ops


def make_probe():
    return {
        'chanMap': np.arange(4, dtype=np.int32),
        'xc': np.array([0, 20, 0, 20], dtype=np.float32),
        'yc': np.array([0, 0, 20, 20], dtype=np.float32),
        'kcoords': np.zeros(4, dtype=np.float32),
        'n_chan': 4,
    }


def make_settings(**overrides):
    settings = DEFAULT_SETTINGS.copy()
    settings.update({'n_chan_bin': 4})
    settings.update(overrides)
    return settings


def test_closed_loop_setting_validation():
    probe = make_probe()
    device = torch.device('cpu')

    with pytest.raises(ValueError):
        initialize_ops(
            make_settings(closed_loop_identity_mode='invalid'),
            probe, 'int16', True, False, device, False
        )

    with pytest.raises(ValueError):
        initialize_ops(
            make_settings(closed_loop_identity_mode='preserve'),
            probe, 'int16', True, False, device, False
        )

    with pytest.raises(ValueError):
        initialize_ops(
            make_settings(closed_loop_nonrigid_nblocks=-1),
            probe, 'int16', True, False, device, False
        )

    ops, settings = initialize_ops(
        make_settings(
            closed_loop_identity_mode='preserve',
            closed_loop_prior_path='fake_prior.npz',
            closed_loop_alignment_mode='nonrigid',
        ),
        probe, 'int16', True, False, device, False
    )
    assert settings['closed_loop_alignment_mode'] == 'nonrigid'


def test_prior_bundle_round_trip(tmp_path):
    ops = {
        'wPCA': torch.eye(5, dtype=torch.float32)[:2],
        'xc': np.array([0, 10, 0], dtype=np.float32),
        'yc': np.array([0, 0, 10], dtype=np.float32),
        'settings': {'closed_loop_identity_mode': 'off'},
        'closed_loop_loop_index': 3,
        'drift_fingerprint_depths': np.array([0, 10, 20], dtype=np.float32),
        'drift_fingerprint': np.ones((3, 2), dtype=np.float32),
        'drift_fingerprint_centered': np.full((3, 2), 2.0, dtype=np.float32),
        'yblk': np.array([0, 20], dtype=np.float32),
    }
    wall = torch.randn(2, 3, 2)
    global_ids = np.array([7, 9], dtype=np.int32)
    is_ref = np.array([1, 0], dtype=np.float32)
    est = np.array([0.01, 0.2], dtype=np.float32)
    prior_path = tmp_path / 'prior_bundle.npz'

    closed_loop.save_prior_bundle(
        prior_path, ops, wall, global_ids, is_ref, est, tmp_path,
        status=np.array(['active', 'stale']),
        source_loop=np.array([0, 1], dtype=np.int32),
    )
    bundle = closed_loop.load_prior_bundle(prior_path)

    assert np.array_equal(bundle['global_unit_ids'], global_ids)
    assert bundle['templates'].shape == (2, 5, 3)
    assert np.array_equal(bundle['source_loop'], np.array([0, 1], dtype=np.int32))
    assert bundle['metadata']['loop_index'] == 3
    assert bundle['fingerprint'].shape == (3, 2)
    assert bundle['fingerprint_centered'].shape == (3, 2)
    assert np.array_equal(bundle['yblk'], np.array([0, 20], dtype=np.float32))


def test_estimate_rigid_shift():
    depths = np.arange(0, 60, 10, dtype=np.float32)
    base = np.column_stack([np.sin(depths / 10), np.cos(depths / 15)]).astype(np.float32)
    shifted = np.roll(base, 2, axis=0)

    shift_um, debug = closed_loop.estimate_rigid_shift(
        depths, shifted, depths, base, binning_depth=10, max_shift_um=30
    )

    assert abs(abs(shift_um) - 20) <= 10
    assert debug['aligned_prior_fingerprint'].shape == base.shape


def test_estimate_nonrigid_shift_falls_back_when_reference_missing():
    depths = np.arange(0, 80, 10, dtype=np.float32)
    prior = np.column_stack([np.sin(depths / 10), np.cos(depths / 14)]).astype(np.float32)
    current = closed_loop._interp_fingerprint(depths, prior, depths, 10.0)
    ops = {
        'yblk': np.array([15, 45], dtype=np.float32),
        'nblocks': 2,
        'binning_depth': 10,
        'settings': {
            'closed_loop_nonrigid_nblocks': 0,
            'closed_loop_nonrigid_max_shift_um': 20.0,
            'closed_loop_nonrigid_smoothing': 1.0,
            'closed_loop_nonrigid_min_score_gain': 0.01,
            'closed_loop_nonrigid_max_residual_um': 30.0,
        },
    }
    prior_bundle = {
        'fingerprint_depths': depths,
        'fingerprint': prior,
    }

    chosen_shift, debug = closed_loop.estimate_nonrigid_shift(
        prior_bundle,
        depths,
        current,
        10.0,
        ops,
    )

    assert debug['used_mode'] == 'rigid'
    assert debug['fallback_reason'] == 'missing_nonrigid_reference'
    assert np.allclose(chosen_shift, 10.0)


def test_estimate_nonrigid_shift_recovers_depth_varying_residual():
    depths = np.arange(0, 120, 10, dtype=np.float32)
    prior = np.column_stack([np.sin(depths / 12), np.cos(depths / 17)]).astype(np.float32)
    true_total = 12.0 + np.linspace(-6, 6, depths.size).astype(np.float32)
    current = closed_loop._interp_fingerprint_with_shifts(depths, prior, depths, true_total)
    rigid_shift, _ = closed_loop.estimate_rigid_shift(
        depths, prior, depths, current, binning_depth=10, max_shift_um=20
    )
    ops = {
        'yblk': np.array([15, 35, 55, 75, 95], dtype=np.float32),
        'nblocks': 3,
        'binning_depth': 10,
        'settings': {
            'closed_loop_nonrigid_nblocks': 3,
            'closed_loop_nonrigid_max_shift_um': 12.0,
            'closed_loop_nonrigid_smoothing': 0.5,
            'closed_loop_nonrigid_min_score_gain': 0.001,
            'closed_loop_nonrigid_max_residual_um': 15.0,
        },
    }
    prior_bundle = {
        'fingerprint_depths': depths,
        'fingerprint': prior,
        'fingerprint_centered': prior - prior.mean(axis=0, keepdims=True),
        'yblk': np.array([15, 35, 55, 75, 95], dtype=np.float32),
    }

    chosen_shift, debug = closed_loop.estimate_nonrigid_shift(
        prior_bundle,
        depths,
        current,
        rigid_shift,
        ops,
    )

    rigid_error = np.mean(np.abs(true_total - rigid_shift))
    recovered = np.interp(depths, ops['yblk'], chosen_shift).astype(np.float32)
    recovered_error = np.mean(np.abs(true_total - recovered))

    assert debug['accepted'] is True
    assert debug['used_mode'] == 'nonrigid'
    assert debug['score_gain'] > 0
    assert recovered_error < rigid_error


def test_estimate_nonrigid_shift_rejects_small_gain():
    depths = np.arange(0, 100, 10, dtype=np.float32)
    prior = np.column_stack([np.sin(depths / 10), np.cos(depths / 15)]).astype(np.float32)
    current = closed_loop._interp_fingerprint(depths, prior, depths, 10.0)
    rigid_shift, _ = closed_loop.estimate_rigid_shift(
        depths, prior, depths, current, binning_depth=10, max_shift_um=20
    )
    ops = {
        'yblk': np.array([20, 50, 80], dtype=np.float32),
        'nblocks': 2,
        'binning_depth': 10,
        'settings': {
            'closed_loop_nonrigid_nblocks': 2,
            'closed_loop_nonrigid_max_shift_um': 10.0,
            'closed_loop_nonrigid_smoothing': 1.0,
            'closed_loop_nonrigid_min_score_gain': 0.05,
            'closed_loop_nonrigid_max_residual_um': 15.0,
        },
    }
    prior_bundle = {
        'fingerprint_depths': depths,
        'fingerprint': prior,
        'fingerprint_centered': prior - prior.mean(axis=0, keepdims=True),
        'yblk': np.array([20, 50, 80], dtype=np.float32),
    }

    chosen_shift, debug = closed_loop.estimate_nonrigid_shift(
        prior_bundle,
        depths,
        current,
        rigid_shift,
        ops,
    )

    assert debug['accepted'] is False
    assert debug['used_mode'] == 'rigid'
    assert debug['fallback_reason'] == 'insufficient_score_gain'
    assert np.allclose(chosen_shift, rigid_shift)


def test_select_matching_prior_mask_prefers_reference_units():
    prior_bundle = {
        'global_unit_ids': np.array([1, 2, 3, 4], dtype=np.int32),
        'status': np.array(['active', 'stale', 'active', 'active']),
        'is_ref': np.array([1, 0, 1, 0], dtype=np.float32),
        'est_contam_rate': np.array([0.01, 0.01, 0.3, 0.05], dtype=np.float32),
    }

    mask = closed_loop.select_matching_prior_mask(prior_bundle)

    assert mask.tolist() == [True, False, True, False]


def test_adapt_carried_templates_respects_min_spikes():
    wPCA = torch.eye(5, dtype=torch.float32)
    aligned_templates = np.ones((2, 5, 3), dtype=np.float32)
    aligned_wall = closed_loop.wall_from_templates(aligned_templates, wPCA)
    discovery_templates = aligned_templates.copy()
    discovery_templates[0] *= 3
    discovery_templates[1] *= 4
    discovery_wall = closed_loop.wall_from_templates(discovery_templates, wPCA)

    templates, adapted_wall, adapted_mask, source = closed_loop.adapt_carried_templates(
        aligned_templates,
        aligned_wall,
        np.array([100, 5], dtype=np.int32),
        {
            0: {'prior_index': 0},
            1: {'prior_index': 1},
        },
        discovery_wall,
        alpha=0.5,
        min_spikes=10,
        wPCA=wPCA,
    )

    assert adapted_mask.tolist() == [True, False]
    assert source.tolist() == [0, -1]
    assert np.allclose(templates[0], 2.0)
    assert np.allclose(templates[1], 1.0)
    assert adapted_wall.shape == aligned_wall.shape


def test_closed_loop_prior_templates_match_template_matching_layout():
    wPCA = torch.eye(5, dtype=torch.float32)[:2]
    templates = np.random.randn(3, 5, 4).astype(np.float32)
    wall = closed_loop.wall_from_templates(templates, wPCA)
    ops = {
        'nt': 5,
        'wPCA': wPCA,
    }

    ctc = template_matching.prepare_matching(ops, wall.transpose(1, 2).contiguous())

    assert wall.shape == (3, 4, 2)
    assert ctc.shape[0] == 3
    assert ctc.shape[1] == 3


def test_warp_templates_nonrigid_returns_valid_shape():
    templates = np.random.randn(2, 5, 4).astype(np.float32)
    probe = make_probe()
    ops = {
        'probe': probe,
        'iKxx': torch.eye(4, dtype=torch.float32),
        'settings': {'sig_interp': 20.0},
    }

    warped = closed_loop.warp_templates_nonrigid(
        templates,
        ops,
        yblk=np.array([0, 20], dtype=np.float32),
        shifts_um=np.array([5, 10], dtype=np.float32),
        device=torch.device('cpu'),
    )

    assert warped.shape == (2, 5, 4)


def test_save_to_phy_closed_loop_exports_global_ids(tmp_path):
    probe = make_probe()
    nt = 5
    n_pcs = 2
    wall = torch.randn(2, 4, n_pcs)
    st = np.array([
        [10, 0, 5],
        [20, 0, 5],
        [30, 1, 6],
        [40, 1, 6],
    ], dtype=np.float64)
    clu = np.array([0, 0, 1, 1], dtype=np.int32)
    tF = torch.randn(4, 3, n_pcs)
    ops = {
        'probe': probe,
        'xc': probe['xc'],
        'yc': probe['yc'],
        'Wrot': torch.eye(4),
        'wPCA': torch.eye(nt, dtype=torch.float32)[:n_pcs],
        'nt': nt,
        'fs': 30000,
        'duplicate_spike_bins': 1,
        'data_dtype': 'int16',
        'settings': {
            'acg_threshold': 0.2,
            'ccg_threshold': 0.25,
            'fs': 30000,
            'n_chan_bin': 4,
            'filename': [tmp_path / 'fake.bin'],
            'nearest_chans': 3,
        },
        'closed_loop_spike_positions': np.array([
            [0, 0], [0, 0], [20, 20], [20, 20]
        ], dtype=np.float32),
        'closed_loop_feature_ind': np.array([
            [0, 1, 2], [1, 2, 3]
        ], dtype=np.uint32),
        'closed_loop_cluster_global_ids': np.array([10, 11], dtype=np.int32),
        'closed_loop_spike_global_ids': np.array([10, 10, 11, 11], dtype=np.int32),
    }

    io.save_to_phy(
        st, clu, tF, wall, probe, ops, imin=0, results_dir=tmp_path
    )

    assert (tmp_path / 'cluster_global_ids.npy').is_file()
    assert (tmp_path / 'spike_global_ids.npy').is_file()
    assert np.array_equal(
        np.load(tmp_path / 'cluster_global_ids.npy'),
        np.array([10, 11], dtype=np.int32),
    )
    assert np.array_equal(
        np.load(tmp_path / 'spike_global_ids.npy'),
        np.array([10, 10, 11, 11], dtype=np.int32),
    )


def test_save_cross_loop_outputs(tmp_path):
    ops = {
        'closed_loop_map_rows': [
            {
                'local_cluster_id': 0,
                'global_unit_id': 5,
                'source_loop': 1,
                'status': 'carried',
                'match_confidence': 1.0,
                'adapted': True,
                'n_spikes_matched': 20,
                'duplicate_similarity': 0.9,
                'duplicate_overlap': 0.2,
            }
        ],
        'closed_loop_summary': {'enabled': True, 'n_carried': 1},
        'closed_loop_debug': {'candidate_shifts_um': np.array([0, 10], dtype=np.float32)},
    }
    closed_loop.save_cross_loop_outputs(tmp_path, ops)
    assert (tmp_path / 'cross_loop_unit_map.csv').is_file()
    assert (tmp_path / 'cross_loop_summary.json').is_file()
    assert (tmp_path / 'cross_loop_debug.npz').is_file()
