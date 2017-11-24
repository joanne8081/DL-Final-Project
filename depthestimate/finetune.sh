#/bin/bash
DATAFILE=$1
DUMPFILE=$2
PKLFILE=$3
PSFILE=$4
BN0=$5
EPOCH=$6
RATE=$7
TESTSIZE=$8
PKLFILE2="${PKLFILE}/twobranch_nn.pkl"

echo =================================Epoch 0===================================
xvfb-run -a python nn.py data=$DATAFILE dump=$DUMPFILE pkl=../demo/twobranch_v2.pkl lr=$7 bno=$5 finetune >> garbage.txt
for number in `seq 1 $6`; 
do
echo =================================Epoch $number===================================
value=`echo $7 | sed -e 's/[eE]+*/\\*10\\^/'`
learning_rate=`echo "$value*((0.99)^$number)" | bc -l`
xvfb-run -a python nn.py model=$DUMPFILE dump=$PKLFILE lr=$learning_rate exportpkl >> garbage.txt
xvfb-run -a python nn.py data=$DATAFILE dump=$DUMPFILE pkl=$PKLFILE2 bno=$5 lr=$learning_rate finetune >> garbage.txt
tarn=$((number%5))
if [ "$tarn" -eq 0 ] || [ $number -eq $6 ]; then
	TARFILE="${PSFILE}_ep${number}.tar.gz"
	PSFILE2="${PSFILE}_cp"
	mkdir $PSFILE2
	echo Do Test and Copy to $PSFILE2 and Tar to $TARFILE
	xvfb-run -a python nn.py data=$DATAFILE model=$DUMPFILE dump=$PSFILE num=$8
	testsize=`echo "$8 - 1" | bc`
	echo $textsize
	for n in `seq 0 $testsize`;
	do
		for num in `seq 1 6 29`;
		do
		cmd="cp ${PSFILE}/pts_${n}_${num}.txt $PSFILE2"
		$cmd
		done
	done
	tar zcvf $TARFILE $PSFILE2 >> garbage.txt
	rm -rf $PSFILE2
fi
rm -rf garbage.txt
done
