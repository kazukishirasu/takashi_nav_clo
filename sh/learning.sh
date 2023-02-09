for i in `seq 1 10`
do
    echo "$i"
    rosrun nav_cloning add_learning.py "$i"
    sleep 10s
done

