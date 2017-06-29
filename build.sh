if ! [ -f ./data/dataset/news.2011.en.shuffled ]; then
	mkdir ./data
	mkdir ./data/temp
	mkdir ./data/dataset

	if ! [ -f ./data/temp/training-monolingual.tgz ]; then
		curl -o ./data/temp/training-monolingual.tgz http://statmt.org/wmt11/training-monolingual.tgz
	fi

	tar -xzvf ./data/temp/training-monolingual.tgz -C data/temp -k

	cp -n ./data/temp/training-monolingual/news.2011.en.shuffled ./data/dataset/news.2011.en.shuffled
fi
   
if [ "$1" = "gpu" ]; then
	echo "Building GPU docker image"
	nvidia-docker build -f ./Dockerfile_gpu -t deepspell-gpu .
else
	echo "Building CPU docker image"
	docker build -f ./Dockerfile_cpu -t deepspell-cpu .
fi
