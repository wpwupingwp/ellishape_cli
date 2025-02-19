# cd src/data
echo "single mode"
$a=ls *.png
get-date -format "HH:mm:ss.fff"
foreach ($i in $a)
{
    &rye run python ../ellishape_cli/cli.py -i $i > log.txt;
}
get-date -format "HH:mm:ss.fff"
echo "batch mode"
get-date -format "HH:mm:ss.fff"
&rye run python ../ellishape_cli/cli.py -I list.txt > log.txt
get-date -format "HH:mm:ss.fff"
