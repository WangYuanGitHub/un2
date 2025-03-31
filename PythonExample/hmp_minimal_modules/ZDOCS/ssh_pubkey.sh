key_name=home2
mkdir -p ./TEMP
wget --user=*******  --password=PASSWORD_FOR_NEXTCLOUD   http://cloud.*******.top:4080/remote.php/dav/files/*******/keys/$key_name.pub -O ./TEMP/_xkey
mkdir -p  ~/.ssh/
cat  ./TEMP/_xkey >>  ~/.ssh/authorized_keys 
cat  ~/.ssh/authorized_keys
