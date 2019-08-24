#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "current work direction is:"
pwd

if [ $# != 4 ]
then
	echo "Four parameters are needed. Run the command as: bash trainAndClassify.sh --path_to_scratch_folder --path_to_trainingFileList.txt 
--path_to_testFileList.txt --path_to_outputListFile.txt "
	exit 1
fi

cd $DIR
echo "### current work direction is:"
pwd
python --version
python extractScatFileName.py
python class_reader_loader.py --pathOfTrainFileList $2 --pathOfTestFileList $3
echo 1
python write_to_tfrecords.py --pathOfScratchFolder $1 --pathOfTrainFileList $2 --pathOfTestFileList $3
echo 2

python training.py --pathOfScratchFolder $1
echo 3
python test.py --pathOfScratchFolder $1 --pathOfTestFileList $3 --pathOfOutput $4
echo 4


# bash trainAndClassify.sh ../scratch_folder ../MIREX2018_v1/debug_data/trainListFile.txt ../MIREX2018_v1/debug_data/testListFile.txt ../MIREX2018_v1/debug_data/outputListFile.txt
