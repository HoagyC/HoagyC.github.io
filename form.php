<?php
$myfile = fopen("content.csv", "a+") or die("Unable to open file!");
$string = $_POST['name'].','.$_POST['location'].',['.$_POST['tags'].'],['.$_POST['reviews'].'].';
fwrite($myfile);
fclose($myfile);
?>