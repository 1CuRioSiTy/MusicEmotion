#!/usr/bin/env bash
start_feature_extraction(){
	DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
	echo $DIR
	tar -zxf preprocessing/ScatNet.tar.gz -C preprocessing/
	if [ $? == 0 ]
	then
		echo "ScatNet decompressed successfully."
	else
		echo "Please enter folder <MIREX2018>, then run the command: bash extractFeatures.sh --path_to_scratch_folder --path_to_featureExtractionListFile.txt"
	fi

	if [ $# != 2 ]
	then
		echo "two parameters are needed. Run the command as: bash extractFeatures.sh --path_to_scratch_folder --path_to_featureExtractionListFile.txt"
		exit 1
	fi

	cd $DIR/preprocessing/ScatNet/
	echo "### current work direction is:"
	pwd
	echo "### path to featureExtractionListFile:"
	echo $1
	echo "### path to feature extraction results:"
	echo $1/preprocessing/scat_coefficients/
	echo "### path to tfrecords:"
	echo $1/preprocessing/data_tfrecords/

	TEMP_FILE=$1"/temp_line.txt"
	if [ -f "$TEMP_FILE" ]
	then
		current_line_in_files=$(<$TEMP_FILE)
	
	else
		current_line_in_files=0
		cd $DIR
		touch $TEMP_FILE
		chmod 777 $TEMP_FILE
		echo $current_line_in_files > $TEMP_FILE
	fi
	TOTAL_L=$( cat $2 | egrep -v '^\s*$' | wc -l )
	TOTAL_L=$((TOTAL_L-1))
	echo 'total line of files(from 0 to the end): '$TOTAL_L
	echo 'current line of files: '$current_line_in_files


	matlab -nodesktop -nosplash -r "addpath_scatnet;scattering_transform_all('$1','$2','$current_line_in_files','$TEMP_FILE');exit;"


}

set -e
trap 'echo "Restart from Segmentation Violation Error"; start_feature_extraction $1 $2' SIGSEGV
trap 'case $? in
        139) echo "Restart from Segmentation Violation Error"; start_feature_extraction $1 $2;
      esac' EXIT

TEMP_FILE=$1"/temp_line.txt"
if [ -f "$TEMP_FILE" ]
then
	current_line_in_files=$(<$TEMP_FILE)

else
	current_line_in_files=0
	touch $TEMP_FILE
	chmod 777 $TEMP_FILE
	echo $current_line_in_files > $TEMP_FILE
fi
TOTAL_L=$( cat $2 | egrep -v '^\s*$' | wc -l )
TOTAL_L=$((TOTAL_L-1))

if [ $TOTAL_L == $current_line_in_files ]
then
	echo 'total line of files(from 0 to the end): '$TOTAL_L
	echo 'current line of files: '$current_line_in_files

	echo "### feature extraction has been finished. Results are in folder $1/preprocessing/scat_coefficients/"
else
	start_feature_extraction $1 $2
	echo "### feature extraction has been finished. Results are in folder $1/preprocessing/scat_coefficients/"
fi

# e.g. bash extractFeatures.sh ../scratch_folder ../MIREX2018_v1/debug_data/featureExtractionListFile.txt



