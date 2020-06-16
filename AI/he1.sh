

model_address="../../data/result/my_model.CNN_3test" 
labels_address="/data2/ben/HE/data/label/reg-tmb.csv"


python model.py --save_model_address $model_address  \
--cross_validation_method 1 \
--label_name label \
--scan_window_suffix *color_norma.png   \
--labels_address $labels_address
#--scan_window_suffix *.orig.png   \


#python draw_heatmap.py --step_x 50 --step_y 50 \
#--save_model_address $model_address \
#--image_address ../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs
