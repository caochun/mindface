这是在Linux环境下安装ES数据库服务器的教程，这里给出两种方法：Docker直装和自行配置。

下面本教程将分别介绍这两种方法，并对可能遇到的问题给出解决方案。

# 方法一 DOCKER直装
请注意，这一方法仅能用作开发用途，因为它并不安全，如果需要更好的部署选项，请根据自行配置部分并关注elastic官网给出的部署策略进行更多优化：https://www.elastic.co/docs/get-started/deployment-optio
## docker-desktop 的安装
```bash
sudo apt-get install ./docker-desktop-amd64.deb
```

通常，运行后会出现以下报错：
```
......
E: Package 'docker-ce-cli' has no installation candidate
......
```

解决方法为：
```bash
# 安装阿里云gpg证书
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add 
 
# 在指定目录下新建docker.list文件，添加阿里云镜像源
cd /etc/apt/sources.list.d        
sudo touch docker.list            
sudo chmod 666 docker.list
sudo echo "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" > docker.list
```

之后update、upgrade并执行安装即可：
```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt install docker-ce docker-ce-cli containerd.io
sudo apt-get install ./docker-desktop-amd64.deb
```
随后终端执行 `docker --version` 以确认docker是否完成安装

## docker 配置
完成docker的安装后，我们尝试启动docker并拉取hello-world：
```bash
systemctl --user start docker-desktop
sudo docker pull hello-world
```
这时通常会报以下错误：
```
Using default tag: latest
Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
```

为了解决类似的问题，我们首先需要为docker配置镜像站：
```bash
cd /etc/docker/
sudo vim daemon.json
sudo systemctl restart docker
```
daemon.json中的内容如下例所示：
```json
{
  "registry-mirrors" : [
    "https://docker.registry.cyou",
    "https://docker-cf.registry.cyou",
    "https://dockercf.jsdelivr.fyi",
    "https://docker.jsdelivr.fyi",
    "https://dockertest.jsdelivr.fyi",
    "https://mirror.aliyuncs.com",
    "https://dockerproxy.com",
    "https://mirror.baidubce.com",
    "https://docker.m.daocloud.io",
    "https://docker.nju.edu.cn",
    "https://docker.mirrors.sjtug.sjtu.edu.cn",
    "https://docker.mirrors.ustc.edu.cn",
    "https://mirror.iscas.ac.cn",
    "https://docker.rainbond.cc",
    "https://do.nark.eu.org",
    "https://dc.j8.work",
    "https://dockerproxy.com",
    "https://gst6rzl9.mirror.aliyuncs.com",
    "https://registry.docker-cn.com",
    "http://hub-mirror.c.163.com",
    "http://mirrors.ustc.edu.cn/",
    "https://mirrors.tuna.tsinghua.edu.cn/",
    "http://mirrors.sohu.com/"
  ],
  "insecure-registries" : [
    "registry.docker-cn.com",
    "docker.mirrors.ustc.edu.cn"
  ],
  "debug": true,
  "experimental": false
}
```
可以通过以下命令验证是否配置成功：
```bash
sudo docker info
sudo systemctl status docker.service
```
第一条命令中能正确看到`Registry Mirrors`字段，第二条命令中能正确看到`Active`字段为`active(running)`

此时重新拉取`sudo docker pull hello-world`，可得到以下结果：
```
Using default tag: latest
latest: Pulling from library/hello-world
e6590344b1a5: Pull complete 
Digest: sha256:ec153840d1e635ac434fab5e377081f17e0e15afab27beb3f726c3265039cfff
Status: Downloaded newer image for hello-world:latest
docker.io/library/hello-world:latest
```
并运行 `sudo docker run hello-world`，得到以下结果即说明配置完成：
```
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

## elasticsearch快速配置
elastic官方给Docker Desktop提供了快速搭建的指令，只有一步即可：
```bash
curl -fsSL https://elastic.co/start-local | sudo sh
```
我们也可以手动下载start-local脚本，并直接运行脚本：
```bash
sudo bash start-local.sh
```

运行完成后会给出账号和密码，可以将其添加到ES_USERNAME和ES_PASSWORD环境变量中并source。账号和密码以及其他配置也可以在./elastic-start-local/.env中找到。
如果要启动es，运行./elastic-start-local/start.sh；如果要关闭es，运行./elastic-start-local/stop.sh；如果要卸载es，运行./elastic-start-local/uninstall.sh。

9200是es的端口，5601是kinaba的端口，可以通过`open http://localhost:port`访问具体的端口，并使用账号密码登录。

至此，方法一docker直装已经完成es的安装。请注意按照官方文档这是仅用于开发的方法，需求更安全的服务器请在方法二自行配置和官方文档的基础上进行额外设置。

# 方法二 自行配置
目前先给出elastic官网给出的教程，后续将进行补充。
https://www.elastic.co/docs/get-started/deployment-optio