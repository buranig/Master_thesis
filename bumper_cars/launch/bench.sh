#! /bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Function to run simulations
run_simulations() {
    local timeout_duration=10s
    local carNumber=$1
    local predHorizon=$2
    local desiredThrottle=$3
    local model=$4
    local alg=$5
    local scenario=$6

    # Use sed to update parameters
    sed -i "s/\(^ *ph: \).*/\1$predHorizon/" "../config/controller.yaml"

    if [ "$model" == "pacejka" ] || [ "$model" == "dynamic" ]; then
        sed -i "s/\(^ *model_type: \).*/dynamic/" "../config/carModel.yaml"
        if [ "$model" == "pacejka" ]; then
            sed -i "s/\(^ *tire_model: \).*/pacejka/" "../config/carModel.yaml"
        elif [ "$model" == "dynamic" ]; then
            sed -i "s/\(^ *tire_model: \).*/dynamic/" "../config/carModel.yaml"
        fi
    else
        sed -i "s/\(^ *model_type: \).*/\1$model/" "../config/carModel.yaml"
    fi
    
    # Echo and run commands with the current carNumber
    timeout "$timeout_duration" ros2 launch bumper_cars collision-avoidance.launch.py carNumber:="$carNumber" static_throttle:="$desiredThrottle" alg:="$alg" write_csv:=True> /dev/null 2>&1 &
    timeout "$timeout_duration" ros2 launch lar_utils simulate.launch.py carNumber:="$carNumber" scenario:="$scenario" > /dev/null 2>&1 &

    # Wait for all background processes to complete
    wait
}



# Remove old data
rm -rf ../bumper_cars/csv/*
# touch "$parent_path/../csv/all_data.csv"
echo "carNumber, predHorizon, desiredThrottle, alg, model, scenario, cum_time, lap_number, it_time, isd" > "../bumper_cars/csv/all_data.csv"


# Establish variables for which to test for

# carNumber
carNumS=1
carNumE=1

# predHorizon
phS=0.5
phE=0.5
predHor_res=0.25
predHors=$(seq $phS $predHor_res $phE)

# desiredThrottle
dsS=0.1
dsE=0.1
desThrottle_res=0.1
desThrottles=$(seq $dsS $desThrottle_res $dse)

# alg
algs=("dwa" "cbf" "c3bf" "lbp" "mpc" "mpc_gpu")
algs=("dwa")
# model (For DWA, LBP, CBF and C3BF)
models=("kinematic" "dynamic" "pacejka")
models=("kinematic")
# scenario
scenariosS=0
scenariosE=1

total_iterations=$(( ($carNumE - $carNumS + 1) * ${#predHors[@]} * ${#desThrottles[@]} * ${#algs[@]} * ${#models[@]} * ($scenariosE - $scenariosS + 1) ))
current_iteration=0

echo -ne '                      (0%)\r'
# Loop through each carNumber for parallel execution
for (( carNumber=$carNumS; carNumber<=$carNumE; carNumber++ ))
do
    for predHor in "${predHors[@]}"
    do
        for desThrottle in "${desThrottles[@]}"
        do
            for alg in "${algs[@]}"
            do
                for model in "${models[@]}" 
                do
                    for (( scenario=$scenariosS; scenario<=$scenariosE; scenario++ ))
                    do
                        progress=$((current_iteration * 100 / total_iterations))
                        echo -ne "["
                        for ((i=0; i<progress; i+=2)); do echo -n "#"; done
                        for ((i=progress; i<100; i+=2)); do echo -n " "; done
                        echo -ne "] ($progress%)\r"

                        ((current_iteration++))
                        run_simulations "$carNumber" "$predHor"  "$desThrottle" "$alg" "$model" "$scenario"  
                    done
                done
            done
        done
    done
done


cat ../bumper_cars/csv/data*.csv >> ../bumper_cars/csv/all_data.csv

echo -ne '#######################   (100%)\r'
echo -ne '\n'

echo "All simulations completed"