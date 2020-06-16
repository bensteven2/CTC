
#images_dir_root="/data2/ben/HE/data/TCGA/lung/"
images_dir_root="/data2/ben/HE/data/xiehe/cesc/HE_image"
##step 1 prepare tumor ragion data
if false; then
python data_prepare.py  \
--images_dir_root $images_dir_root \
--size_square 512 \
--prepare_types 1
fi



##color nomalization
if false; then
python Stain_Color_Normalization.py \
--log_dir "/data2/ben/HE/data/result/color_logs/" \
--data_dir "/data2/ben/HE/data/TCGA/lung/" \
--tmpl_dir "/data2/ben/HE/data/result/color_tmp/" 
fi

##
##my_model.h5

#labels_address="/data2/ben/HE/data/TCGA/labels/reg-tmb.csv"
#label_name="file_name,labels"


labels_address="../../data/geneis/CTC/result_test6.xlsx"
#labels_address="../../data/geneis/CTC/result.bencreate20200229.xlsx"
#labels_address="../../data/geneis/CTC/result.bencreate20200229_1.xlsx"
#labels_address="../../data/geneis/CTC/result.bak2019.01.15.xlsx"
label_name="file_name,labels"

images_dir="../../data/geneis/CTC/images"
#save_model_address="../../data/geneis/CTC/my_model.CNN_3"
save_model_address="../../data/geneis/CTC/my_model.CNN_5"
size_square=120
#### a good example: 
if false;then
mode="train"
epochs=10
batch_size=2
times=3 #
L1=8 #"the number of the first conv2d
F1=5 # "the size of the first conv2d layer")

F2=2 #"the size of the first maxpooling layer")

L2=8 #"the number of the second conv2d ")
F3=3 #"the size of the second conv2d layer"
cross_validation_method=0
fi


mode="train"

epochs=1
batch_size=2
times=3 #
L1=8 #"the number of the first conv2d
F1=5 # "the size of the first conv2d layer")

F2=2 #"the size of the first maxpooling layer")

L2=8 #"the number of the second conv2d ")
F3=3 #"the size of the second conv2d layer"
cross_validation_method=0
model_type="resnet34"



tensorflow_version=2.0
roc_address=${save_model_address}"_"${mode}"_roc.pdf"
if true;then
python cnn_small_image.py --save_model_address $save_model_address  \
--cross_validation_method $cross_validation_method \
--model $model_type \
--mode $mode \
--label_name $label_name \
--scan_window_suffix *color_norma.png   \
--labels_address $labels_address  \
--n_splits 2  \
--label_types 2 \
--images_dir $images_dir \
--size_square $size_square \
--epochs $epochs \
--times $times \
--L1 $L1 \
--L2 $L2 \
--F1 $F1 \
--F2 $F2 \
--F3 $F3 \
--batch_size $batch_size \
--tensorflow_version $tensorflow_version \
--roc_address $roc_address  \
--roc_title "I am here"
#--scan_window_suffix "*.orig.png"  
fi

