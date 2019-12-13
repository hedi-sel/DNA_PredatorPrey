cd $(dirname $0)

echo "Do you want to generate a clip? (y)" && read doGif &&
if [ $doGif = y ];
then
    convert -delay 10 -loop 0 $(ls -1 plot/$1/*.png | sort -V) clips/${1}.gif;
    echo "Want to save this gif to HediPC? ? (y)" && read doSave &&
    if [ _$doSave = _y ];
    then
        scp clips/${1}.gif hediPC:Documents/Results/${1}.gif;
    fi
    echo "Job finished!"
fi