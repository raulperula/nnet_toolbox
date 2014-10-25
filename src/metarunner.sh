#!/bin/csh
echo "Start"
if ($#argv < 1) then
	echo "Proc arg missing"
else
	unsetenv DISPLAY
	#~ set salida = "$argv[1]_execution_host";
	#~ cat /etc/hostname >> $salida && cat $argv[1] >> $salida
	set date = `date +%F-%H.%M.%S`;
	echo $date;
	set cmd = "nohup /usr/local/bin/matlab -nodisplay < $argv[1] > /tmp/$argv[1]_$date-output &"
	echo $cmd;
	$cmd
endif
echo "End"
