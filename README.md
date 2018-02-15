INSTALLATION
==========

## Hook up the repo
Install git

```
sudo apt-get update
sudo apt-get -y install git
```

Clone the bluegeo repo
Note: replace usr and pwd with your bitbucket username and password

```git clone https://usr:pwd@bitbucket.org/bluegeo/bluegeo.git```

## Install everything
Minimum server requirements can be met using `privision.sh`
Run the script and wait

```
sudo chmod u+x bluegeo/provision.sh
bluegeo/provision.sh
```

## That's it!
