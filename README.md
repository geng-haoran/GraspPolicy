# Grasping Policy

## Env

- GraspNet
  ```shell
    git clone git@github.com:rhett-chen/graspness_implementation.git
    ```
    Then follow this repo install instruction
    ```shell
    conda create -n grasp python==3.8
    # conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    git clone git@github.com:NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    conda install openblas-devel -c anaconda
    # 先改一行code再编译(floor):
    # If you meet the torch.floor error in MinkowskiEngine,
    # you can simply solve it by changing the source code of MinkowskiEngine:
    #MinkowskiEngine/utils/quantization.py 262，
    # from discrete_coordinates =_auto_floor(coordinates) to discrete_coordinates = coordinates
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    # graspness api
    git clone https://github.com/graspnet/graspnetAPI.git
    cd graspnetAPI
    # vim setup.py:  change sklearn -> scikit-learn
    pip install .
    pip install "numpy<1.24"
    pip install pytorch-utils
    ```

    pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib" 