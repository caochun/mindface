这是在Linux环境下安装Milvus的教程，首先给出官方文档网址：https://milvus.io/docs/zh/quickstart.md

官方给出了以下两种在Linux下安装Milvus的方法，主要基于Docker容器。关于Docker的安装，详情请见[ES_Tutorial](./ES_Tutorial.md)。

# 在 Docker 中运行 Milvus (Linux)
官网教程：https://milvus.io/docs/zh/install_standalone-docker.md

安装完docker后，milvus的安装非常简单，只需执行下面的命令即可（详情请查看上面的官方教程）：
```bash
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
# Start the Docker container
$ bash standalone_embed.sh start
```

# 使用 Docker Compose 运行 Milvus (Linux)
官网教程：https://milvus.io/docs/zh/install_standalone-docker-compose.md

如果想在独立部署模式下使用备份，请使用Docker Compose运行Milvus，只需执行下面的命令即可（详情请查看上面的官方教程）：
```bash
# Download the configuration file
$ wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml -O docker-compose.yml
Start Milvus
$ sudo docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```

# 安装 pymilvus 库
```bash
$ pip install -U pymilvus
```