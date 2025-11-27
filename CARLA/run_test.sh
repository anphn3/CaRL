#!/bin/bash
set -e

echo "=========================================="
echo "CaRL Memory Test - CARLA Connected"
echo "=========================================="

# Setup environment
export SCENARIO_RUNNER_ROOT=/home/anphn3/CaRL/CARLA/custom_leaderboard/leaderboard
export LEADERBOARD_ROOT=/home/anphn3/CaRL/CARLA/custom_leaderboard/scenario_runner
export CARLA_ROOT=/home/anphn3/CARLA_0.9.15

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# Create test dir
TEST_DIR="./test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "Test directory: $TEST_DIR"
echo ""

# Monitor GPU in background
{
    echo "timestamp,used_mb,peak_mb"
    max_mem=0
    while true; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if (( $(echo "$mem > $max_mem" | bc -l) )); then
            max_mem=$mem
        fi
        echo "$(date +%s),$mem,$max_mem"
        printf "\r[%s] GPU Memory: %5s MB (Peak: %5s MB)" \
            "$(date +%H:%M:%S)" "$mem" "$max_mem"
        sleep 1
    done
} > "$TEST_DIR/gpu.csv" &
MONITOR_PID=$!

cleanup() {
    echo ""
    echo ""
    kill $MONITOR_PID 2>/dev/null
    
    # Get peak memory
    peak=$(tail -1 "$TEST_DIR/gpu.csv" | cut -d',' -f3)
    
    echo "=========================================="
    echo "Peak GPU Memory: ${peak} MB"
    echo ""
    
    # Estimate for 1024/256
    est_conservative=$(awk "BEGIN {printf \"%.0f\", $peak * 32}")
    est_optimistic=$(awk "BEGIN {printf \"%.0f\", $peak * 32 * 0.7}")
    
    echo "Estimated for batch 1024/256:"
    echo "  Conservative: ${est_conservative} MB ($(awk "BEGIN {printf \"%.1f\", $est_conservative/1024}") GB)"
    echo "  Optimistic:   ${est_optimistic} MB ($(awk "BEGIN {printf \"%.1f\", $est_optimistic/1024}") GB)"
    echo ""
    
    if [ $est_conservative -lt 20480 ]; then
        echo "✅ RTX 3090/4090 (24GB) should work"
    elif [ $est_conservative -lt 40960 ]; then
        echo "⚠️  RTX 3090/4090 (24GB) risky"
        echo "✅ A6000 (48GB) recommended"
    else
        echo "❌ Need A6000 (48GB) or A100 (80GB)"
    fi
    echo "=========================================="
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "Starting training (2 minutes)..."
echo "Press Ctrl+C to stop anytime"
echo ""

timeout 120 python -u train_parallel.py \
    --train_cpp 0 \
    --team_code_folder /home/anphn3/CaRL/CARLA/team_code \
    --num_nodes 1 --node_id 0 \
    --rdzv_addr 127.0.0.1 --rdzv_port 0 \
    --collect_device gpu \
    --git_root /home/anphn3/CaRL/CARLA \
    --carla_root /home/anphn3/CARLA_0.9.15 \
    --exp_name "MemTest" \
    --num_envs_per_gpu 1 \
    --seed 9900 \
    --start_port 100024 \
    --gpu_ids 0 \
    --train_towns 1 \
    --num_envs_per_node 1 \
    --total_batch_size 4 \
    --total_minibatch_size 8 \
    --update_epochs 3 \
    --gamma 0.99 --gae_lambda 0.95 \
    --clip_coef 0.1 --max_grad_norm 0.5 \
    --learning_rate 0.00025 \
    --total_timesteps 500 \
    --lr_schedule linear \
    --use_speed_limit_as_max_speed 0 \
    --beta_min_a_b_value 1.0 \
    --use_new_bev_obs 1 \
    --reward_type simple_reward \
    --consider_tl 1 --eval_time 1200 \
    --terminal_reward 0.0 --normalize_rewards 0 \
    --speeding_infraction 1 \
    --min_thresh_lat_dist 2.0 \
    --map_folder maps_2ppm_cv \
    --pixels_per_meter 2 --route_width 6 \
    --num_route_points_rendered 150 \
    --use_green_wave 0 \
    --image_encoder resnet50 \
    --use_layer_norm 1 \
    --use_vehicle_close_penalty 0 \
    --routes_folder 1000_meters_old_scenarios_01 \
    --render_green_tl 1 --distribution beta \
    --use_termination_hint 1 --use_perc_progress 1 \
    --use_min_speed_infraction 0 \
    --use_leave_route_done 0 \
    --use_layer_norm_policy_head 1 \
    --obs_num_measurements 8 \
    --use_extra_control_inputs 0 \
    --condition_outside_junction 0 \
    --use_outside_route_lanes 1 \
    --use_max_change_penalty 0 \
    --terminal_hint 1.0 --use_target_point 0 \
    --use_value_measurements 1 \
    --bev_semantics_width 192 \
    --bev_semantics_height 192 \
    --pixels_ev_to_bottom 100 \
    --use_history 0 --obs_num_channels 10 \
    --use_off_road_term 1 \
    --beta_1 0.9 --beta_2 0.999 \
    --route_repetitions 20 \
    --render_speed_lines 1 \
    --use_new_stop_sign_detector 1 \
    --use_positional_encoding 0 --use_ttc 1 \
    --num_value_measurements 10 \
    --render_yellow_time 1 \
    --penalize_yellow_light 0 \
    --use_comfort_infraction 1 \
    --use_single_reward 1 \
    --off_road_term_perc 0.95 \
    --render_shoulder 0 --use_shoulder_channel 1 \
    --use_rl_termination_hint 1 \
    --lane_distance_violation_threshold 0.0 \
    --lane_dist_penalty_softener 1.0 \
    --comfort_penalty_factor 0.5 \
    --use_survival_reward 0 \
    --use_exploration_suggest 0 --track 1 \
    --use_temperature False --use_rpo False \
    --rpo_alpha 0.5 \
    --use_hl_gauss_value_loss False \
    --use_lstm False > "$TEST_DIR/train.log" 2>&1

echo ""
echo "✅ Test completed"
exit 0
