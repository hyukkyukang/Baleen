# Below script follows the installation guide of faiss (https://github.com/facebookresearch/faiss)

git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON
make -C build -j faiss
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)
echo "Install faiss successfully!"