#! /bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Function to run simulations
run_simulations() {
    local carNumber=$1
    local predHorizon=$2
    local desiredThrottle=$3
    local model=$4
    local alg=$5
    local scenario=$6
    local gen_traj=$7
    local timeout_duration=$8

    # Use sed to update parameters in config files

    sed -i -e "s/^  ph: .*/  ph: ""$predHorizon/" "../config/controller.yaml"

    if [[ "$model" == "pacejka" || "$model" == "dynamic" ]]; then
        sed -i -e "s/^ *model_type: .*/model_type: 'dynamic'/" "../config/carModel.yaml"
        if [ "$model" == "pacejka" ]; then
            sed -i -e "s/^ *tireModel: .*/tireModel: 'pacejka'/" "../config/carModel.yaml"
        else
            sed -i -e "s/^ *tireModel: .*/tireModel: 'linear'/" "../config/carModel.yaml"
        fi
    else
        sed -i -e "s/^ *model_type: .*/model_type: '$model'/" "../config/carModel.yaml"
    fi

    # Run ROS2 launch commands with timeout
    timeout "$timeout_duration" ros2 launch bumper_cars collision-avoidance.launch.py \
        carNumber:="$carNumber" static_throttle:="$desiredThrottle" alg:="$alg" \
        write_csv:=True gen_traj:="$gen_traj" > /dev/null 2>&1 &
    
    timeout "$timeout_duration" ros2 launch lar_utils simulate.launch.py \
        carNumber:="$carNumber" scenario:="$scenario" > /dev/null 2>&1 &

    # Wait for background processes to complete
    wait
}

#######################################################################
####################### Start of the program ##########################
#######################################################################


# Remove old data
rm -rf ../csv/*
echo "carNumber,predHorizon,desiredThrottle,alg,model,scenario,iteration,car_i,cum_time,lap_time,lap_number,it_time,isd" > "../csv/all_data.csv"

# Establish variables for which to test for
timeout_duration=120s
timeout_generation=140s

# carNumber
carNumS=2
carNumE=5

# predHorizon
phS=0.5
phE=0.5
predHor_res=0.25
predHors=($(LC_ALL=C seq $phS $predHor_res $phE))

# desiredThrottle
dsS=0.2
dsE=0.2
desThrottle_res=0.1
desThrottles=($(LC_ALL=C seq $dsS $desThrottle_res $dsE))

# alg
algs=("dwa" "cbf" "c3bf" "lbp" "mpc" "mpc_gpu")
algs=("dwa" "cbf" "c3bf" "lbp" "mpc_gpu")
# models
models=("kinematic" "dynamic" "pacejka")
models=("kinematic")
# scenario
scenariosS=0
scenariosE=0


repetitions=5

total_iterations=0

# Precompute the total number of iterations the hard way

for alg in ${algs[@]}
do
    for model in ${models[@]}
    do
        for predHor in ${predHors[@]}
        do
            for desThrottle in ${desThrottles[@]}
            do
                for (( carNumber=$carNumS; carNumber<=$carNumE; carNumber++ ))
                do
                    for (( scenario=$scenariosS; scenario<=$scenariosE; scenario++ ))
                    do
                        for (( i=0; i<$repetitions; i++ ))
                        do
                            ((total_iterations++))
                        done
                    done
                done
            done
            # CBF and C3BF don't need to be tested with different prediction horizons
            if [[ $alg == "cbf" || "$alg" == "c3bf" ]]; then
                break
            fi
        done
        # CBF and C3BF don't need to be tested with different models
        # MPPI and LBP are only implemented with kinematic
        if [[ $alg == "cbf" || "$alg" == "c3bf" || "$alg" == "mpc_gpu" || "$alg" == "LBP" ]]; then
                break
        fi
    done
done



current_iteration=0
echo -ne '                      (0%)\r'
# Loop through each carNumber for parallel execution
for alg in ${algs[@]}
do
    for model in ${models[@]}
    do
        for predHor in ${predHors[@]}
        do
            if [[ $alg == "dwa" || "$alg" == "lbp" ]]; then
                genTraj="False"
            fi

            for desThrottle in ${desThrottles[@]}
            do
                for (( carNumber=$carNumS; carNumber<=$carNumE; carNumber++ ))
                do
                    for (( scenario=$scenariosS; scenario<=$scenariosE; scenario++ ))
                    do
                        for (( j=0; j<$repetitions; j++ ))
                        do
                            progress=$((current_iteration * 100 / total_iterations))
                            echo -ne "["
                            for ((i=0; i<progress; i+=2)); do echo -n "#"; done
                            for ((i=progress; i<100; i+=2)); do echo -n " "; done
                            echo -ne "] ($progress%)\r"
                        
                            ((current_iteration++))
                            if [[ $genTraj == "True" ]]; then
                                run_simulations "$carNumber" "$predHor"  "$desThrottle" "$model" "$alg" "$scenario"  "$genTraj" "$timeout_generation"
                                genTraj="False"
                            else
                                run_simulations "$carNumber" "$predHor"  "$desThrottle" "$model" "$alg" "$scenario"  "$genTraj" "$timeout_duration"
                            fi

                            # Add simulation info and store in general file
                            prefix="${carNumber},${predHor},${desThrottle},${alg},${model},${scenario},${j},"
                            find ../csv/ -type f -name 'data*.csv' -exec sed -i -e "s/^/$prefix/" {} \;
                            # sed -i -e "s/^/$prefix/" ../csv/data*.csv
                            cat ../csv/data*.csv >> ../csv/all_data.csv
                        done
                    done
                done
            done
            # CBF and C3BF don't need to be tested with different prediction horizons
            if [[ $alg == "cbf" || "$alg" == "c3bf" ]]; then
                break
            fi
        done
        # CBF and C3BF don't need to be tested with different models
        # MPPI and LBP are only implemented with kinematic
        if [[ $alg == "cbf" || "$alg" == "c3bf" || "$alg" == "mpc_gpu" || "$alg" == "LBP" ]]; then
                break
        fi
    done
done

echo -ne '#######################   (100%)\r'
echo -ne '\n'

echo "All simulations completed"