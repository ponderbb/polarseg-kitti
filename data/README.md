Download the [SemanticKITTI dataset](http://www.semantic-kitti.org/dataset.html#overview) and extract the [velodyne pointclouds](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip) and [odometry labels](www.semantic-kitti.org/assets/data_odometry_labels.zip) in the following manner:

```
./
└── data/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	# Unzip from KITTI Odometry Benchmark Velodyne point clouds.
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 	# Unzip from SemanticKITTI label data.
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── ...
        └── 21/
	    └── ...
```

Citation for the KITTI Vision Benchmark:

```
@inproceedings{geiger2012cvpr,
  author = {A. Geiger and P. Lenz and R. Urtasun},
  title = {{Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite}},
  booktitle = {Proc.~of the IEEE Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3354--3361},
  year = {2012}
}
```