
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

##--data_dir "/data2/practice/yezx/cecs/" \
##
##my_model.h5

#labels_address="/data2/ben/HE/data/TCGA/labels/reg-tmb.csv"
#label_name="file_name,labels"
if false;then
labels_address="/data2/ben/HE/data/TCGA/labels/labels_immune1.csv"
#label_name="patients,POLE"
#label_name="patients,POLD1"
#label_name="patients,PBRM1"
#label_name="patients,STK11"
#label_name="patients,DNMT3A"
#label_name="patients,EGFR"
#label_name="patients,TP53"
#label_name="patients,KRAS"
#label_name="patients,CD274"




labels_address="/data2/ben/HE/data/TCGA/labels/labels_target1.csv"
#label_name="patients,AKT1"
#label_name="patients,ALK"
#label_name="patients,BRAF"
#EGFR,
#label_name="patients,FGFR1"
#label_name="patients,FGFR2"
#label_name="patients,HRAS"
#,KRAS,
#label_name="patients,MET"
#label_name="patients,NRAS"
#label_name="patients,NTRK1"
#label_name="patients,NTRK2"
#label_name="patients,NTRK3"
#label_name="patients,PIK3CA"
#label_name="patients,RET"
#label_name="patients,ROS1"
#label_name="patients,TSC1"
fi


labels_address="/data2/ben/HE/data/TCGA/labels/reg-tmb.csv"
label_name="file_name,labels"

model_address="../../data/result/my_model.CNN_3_reg-tmb20200105" 
############model_address="../../data/result/my_model.CNN_3_20191210--" 

if false;then
python model.py --save_model_address $model_address  \
--cross_validation_method 1 \
--label_name $label_name \
--scan_window_suffix *color_norma.png   \
--labels_address $labels_address  \
--n_splits 2  \
--label_types 2
#--scan_window_suffix "*.orig.png"  
fi

if false;then
step_x=100
step_y=100
begin_x=43000
begin_y=49000
dimensions_x=10000
dimensions_y=5000
images_address=../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs
fi


step_x=100
step_y=100
begin_x=0
begin_y=0
dimensions_x=-1
dimensions_y=-1
#images_address=../../data/geneis/lung/79807-14604-40x001.png
images_address=../../data/geneis/lung/79807-14604-40x002.png



if true;then
python draw_heatmap.py --step_x $step_x --step_y $step_y \
--save_model_address $model_address \
--begin_x $begin_x \
--begin_y $begin_y \
--dimensions_x  $dimensions_x \
--dimensions_y $dimensions_y \
--images_address $images_address 
fi




######################################################for test start
if false;then
python draw_heatmap.nocolornorm.py --step_x 500 --step_y 500 \
--save_model_address $model_address \
--begin_x 43000 \
--begin_y 49000 \
--dimensions_x  10000 \
--dimensions_y 5000 \
--image_address ../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs
fi

if false;then
model_address="../../data/result/my_model.CNN_3test" 
python draw_heatmap.py --step_x 500 --step_y 500 \
--save_model_address $model_address \
--begin_x 30000 \
--begin_y 15000 \
--dimensions_x  10000 \
--dimensions_y 5000 \
--image_address /data2/ben/HE/data/xiehe/cesc/HE_image/S698161/S698161-A1.ndpi
fi


if false;then
model_address="../../data/result/my_model.CNN_3test" 
python draw_heatmap.py --step_x 500 --step_y 500 \
--save_model_address $model_address \
--pattern multiple   \
--scan_window_suffix "*.ndpi" \
--images_dir_root "../../data/xiehe/cesc/HE_image" \
--labels_address "../../data/xiehe/cesc/labels/xiehe.cesc.TMB.class.csv" \
--header_name "Samples,TMB"   \
--ID_prefix_num 7
fi
#####################################################for test end

