My Solutions to Udacity CS 344 Homework
=====

Since I don't own an NVIDA processor. I rented a cloud GPU server from [AWS][1]

## Instructions for setting up a cloud GPU Server on EC2

[Discussion on AWS GPU instance types][2]

Get an EC2 instance that costs less than the c*.4xlarge instance suggested on the Udacity CS 344 course site. Recently AWS made a new class of middle size GPU capable instances available. Launch a g2.2xlarge instance with Ubuntu 12.04:

   * pick Ubuntu Server 12.04 LTS (HVM) 64-bit AMI
   * pick g2.2xlarge 

[Install Fish Shell][7]:
    
    sudo apt-add-repository ppa:fish-shell/release-2
    sudo apt-get update
    sudo apt-get install fish

Now permanently change the shell:

    chsh -s /usr/bin/fish

Add a new user:

    sudo useradd -d /home/user -m user
    sudo passwd user 

Need to add user ‘user’ to sudoers file.

    sudo nano /etc/sudoers

Under user privilege specification append:

    user  ALL=(ALL:ALL) ALL

Add SSH keys for user to instance:

    mkdir ~/.ssh
    cd ~/.ssh/
    touch authorized_keys
    nano authorized_keys 

Now you should be able to ssh into your instance, (*from your local machine*):

     ssh -i ~/.ssh/gpukeypair.pem user@ec2-**-***-**-**.compute-1.amazonaws.com

Once in, add the capability of your instance to [auto-update packages][6] (especially security packages).

   * [**Setup your instance with Open CV, CUDA, etc.**][4]
   * Create an [Elastic IP Address][3]. Contrary to the name, these are static IPs and are FREE. Furthermore, they are assigned to an account, and associated to an instance—all this can be done from the management console. 

###Setup Samba service:
    sudo apt-get install samba
    
Edit the samba config file with:
    
    workgroup = GPU   ...[homes]
        browsable = yes
        writeable = yes

Finish by openning port 445 on AWS under [security groups][5].


[1]: http://aws.amazon.com/
[2]: http://www.kurtsp.com/deep-learning-in-python-with-pylearn2-and-amazon-ec2.html
[3]: http://aws.amazon.com/articles/1346
[4]: https://www.udacity.com/wiki/cs344/ubuntu-dev
[5]: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html
[6]: https://help.ubuntu.com/12.04/serverguide/automatic-updates.html
[7]: http://fishshell.com/
