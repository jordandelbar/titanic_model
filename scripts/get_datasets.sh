$path = "~/titanic/titanic_model/datasets/"
kaggle competitions download -c titanic -p $path
unzip $path/titanic.zip
rm $path/titanic.zip