echo "--------------------------------------------------------------"
echo "-                                                            -"
echo "-These workloads are to be used as performance reference only-"
echo "-     Go the the official MLCommons website to obtain        -"
echo "-          instructions to make an official run              -"
echo "-                                                            -"
echo "--------------------------------------------------------------"

# exit when any command fails
set -e

if [ ! -d "./mlc" ]; then
echo "create virtual environment"
python3 -m venv mlc
fi
. ./mlc/bin/activate && 
pip install mlc-scripts && 
deactivate &&
echo "mlcommon virtual evnironment created & mlcr scripts installed. Done"
