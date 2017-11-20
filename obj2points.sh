OBJFILE=$1
PCFILE=$2

cp $OBJFILE $PCFILE
sed -i.bak '/^v /! d' $PCFILE
sed -i.bak 's/^v //g' $PCFILE
rm $PCFILE.bak
