config = {'stage1_data_path':'/data/lungCT/dsb2017/sample_images',
          'luna_raw':'/data/lungCT/luna/raw',
          'luna_segment':'/data/lungCT/luna/seg-lungs-LUNA16',
          
          'luna_data':'/data/lungCT/luna/temp/rename_data',
          'preprocess_result_path':'/data/lungCT/luna/temp/luna_npy',       
          
          'luna_abbr':'./detector/labels/shorter.csv',
          'luna_label':'./detector/labels/lunaqualified.csv',
          'stage1_annos_path':['./detector/labels/label_job5.csv',
                './detector/labels/label_job4_2.csv',
                './detector/labels/label_job4_1.csv',
                './detector/labels/label_job0.csv',
                './detector/labels/label_qualified.csv'],
          'bbox_path':'../detector/results/res18/bbox/',
          'preprocessing_backend':'python'
         }


#'luna_label':'./detector/labels/lunaqualified.csv',