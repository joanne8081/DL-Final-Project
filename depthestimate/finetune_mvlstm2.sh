#/bin/bash
DATAFILE=$1
DUMPFILE=$2
PKLFILE=$3
PSFILE=$4
BN0=$5
EPOCH=$6
RATE=$7
TESTSIZE=$8
PKLFILE2="${PKLFILE}/twobranch_nnmvlstm2.pkl"

xvfb-run -a python nnmvlstm2.py data=../data dump=finetunelstm2 pkl=../demo/twobranch_v2.pkl lr=3e-5 bno=120  mvfinetune

xvfb-run -a python nnmvlstm2.py model=finetunelstm2 dump=pkldumplstm3 lr=3e-5 exportpkl

xvfb-run -a python nnmvlstm2.py data=../data dump=finetunelstm2 pkl=./pkldumplstm3/twobranch_nnmvlstm2.pkl lr=3e-5 bno=120 mvfinetune

echo =================================Epoch 0===================================
xvfb-run -a python nnmvlstm2.py data=$DATAFILE dump=$DUMPFILE pkl=./pkldumplstm3/twobranch_nnmvlstm2.pkl lr=$7 bno=$5 mvfinetune
for number in `seq 1 $6`; 
do
echo =================================Epoch $number===================================
value=`echo $7 | sed -e 's/[eE]+*/\\*10\\^/'`
learning_rate=`echo "$value*((0.99)^$number)" | bc -l`
xvfb-run -a python nnmvlstm2.py model=$DUMPFILE dump=$PKLFILE lr=$learning_rate exportpkl
xvfb-run -a python nnmvlstm2.py data=$DATAFILE dump=$DUMPFILE pkl=$PKLFILE2 bno=$5 lr=$learning_rate mvfinetune
tarn=$((number%5))
if [ "$tarn" -eq 0 ]; then
	TARFILE="${PSFILE}_ep${number}.tar.gz"
	PSFILE2="${PSFILE}_cp"
	mkdir $PSFILE2
	echo Do Test and Copy to $PSFILE2 and Tar to $TARFILE
	xvfb-run -a python nnmvlstm.py data=$DATAFILE model=$DUMPFILE dump=$PSFILE num=$8 test >> garbage.txt
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

