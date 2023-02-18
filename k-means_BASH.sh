#!/bin/bash

# Initialize variables
data_file="data.txt" # input data file
k=3 # number of clusters
max_iterations=10 # maximum number of iterations
tolerance=0.01 # tolerance for convergence

# Read data from file
data=$(cat $data_file)

# Initialize centroids randomly
centroids=$(echo "$data" | shuf -n $k)

# Loop until convergence or maximum iterations reached
for (( i=1; i<=$max_iterations; i++ ))
do
    # Assign each data point to the closest centroid
    assignments=$(echo "$data" | while read line; do
        closest_centroid=$(echo "$centroids" | awk -v line="$line" '{
            distance = sqrt((($1 - line) ^ 2) + (($2 - line) ^ 2))
            if (NR == 1 || distance < min_distance) {
                min_distance = distance
                closest_centroid = $0
            }
        } END {print closest_centroid}')
        echo "$closest_centroid $line"
    done)

    # Calculate new centroids based on the assigned data points
    new_centroids=$(echo "$assignments" | awk -v k="$k" '{
        x_sum[$1] += $2
        y_sum[$1] += $3
        count[$1] += 1
    } END {
        for (i = 1; i <= k; i++) {
            if (count[i] > 0) {
                x_avg = x_sum[i] / count[i]
                y_avg = y_sum[i] / count[i]
                print x_avg, y_avg
            }
        }
    }')

    # Calculate the change in centroids and check for convergence
    change=$(echo "$centroids" "$new_centroids" | awk -v tolerance="$tolerance" '{
        distance = sqrt((($1 - $3) ^ 2) + (($2 - $4) ^ 2))
        if (distance > tolerance) {
            print "not converged"
            exit
        }
    } END {
        if (NR == 0) {
            print "not converged"
        } else {
            print "converged"
        }
    }')

    # Update centroids if not converged
    if [ "$change" = "not converged" ]
    then
        centroids="$new_centroids"
    else
        break
    fi
done

echo "Final centroids:"
echo "$centroids"
