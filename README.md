# SHINE: A Scalable HNSW Index in Disaggregated Memory

Implementation of a distributed HNSW index for memory disaggregation. 
This is the source code of the paper "SHINE: A Scalable HNSW Index in Disaggregated Memory".

## Setup

### C++ Libraries and Unix Packages

The following C++ libraries and Unix packages are required to compile the code.
Note that `ibverbs` (the RDMA library) is Linux-only. 
The code also compiles without InfiniBand network cards.

* [ibverbs](https://github.com/linux-rdma/rdma-core/tree/master)
* [boost](https://www.boost.org/doc/libs/1_83_0/doc/html/program_options.html) (to support `boost::program_options` for
  CLI parsing)
* pthreads (for multithreading)
* [oneTBB](https://github.com/oneapi-src/oneTBB) (for concurrent data structures)
* a C++ compiler that supports C++20 (we have used `g++-12`)
* cmake
* numactl
* vmtouch (to map index files into main memory)
* axel (a download accelerator for the datasets)

For instance, to install the requirements on Debian, run the following command:
```
apt-get -y install g++ libboost-all-dev libibverbs1 libibverbs-dev numactl cmake libtbb-dev git python3-venv vmtouch axel
```

### Cluster Nodes Configuration

Adjust the IP addresses of the cluster nodes accordingly in `rdma-library/library/utils.cc`:
https://frosch.cosy.sbg.ac.at/mwidmoser/shine-hnsw-index/-/blob/main/rdma-library/library/utils.cc?ref_type=heads#L14-L23

### Compilation

After cloning the repository and installing the requirements, the code must be compiled on all cluster nodes:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This produces two binaries:
- `build/shine`: online memory/compute node service
- `build/shine_offline_builder`: offline hnswlib-based graph builder that exports SHINE shard files

## Download the Data

First, install all Python requirements:
```
cd scripts
python3 -m pip install -r requirements.txt
```

Then, run the following script to download the data. 
This may take a while, we recommend to run the script within a `tmux` session.
Also make sure that `axel` (a download accelerator) is installed.
```
cd data
bash download.sh
```

Finally, create the queries (adjust the `DATASET_PATH` in `create_queries.py`):
```
python3 create_queries.py
```

Now all the data is available in `data/datasets`, move them to a location where all cluster nodes have access to (e.g., to an NFS).
Then, adjust the path in `config.py`.

## Run the Experiments

* TODO

## Offline Build And Online Load

The project now supports building the HNSW graph offline with `hnswlib` and exporting it into SHINE's native
memory-node shard format.

The required `hnswlib` headers are vendored under `thirdparty/hnswlib`, so a normal `git clone` is enough. No
extra submodule initialization is needed.

Build an offline index:
```bash
./scripts/build_offline_index.sh \
  --data-path /path/to/dataset-or-dir \
  --memory-nodes 2 \
  --threads 32 \
  --m 32 \
  --ef-construction 200 \
  --output-prefix /path/to/index/shine_index
```

This writes files like:
```text
/path/to/index/shine_index_node1_of2.dat
/path/to/index/shine_index_node2_of2.dat
/path/to/index/shine_index.meta.json
```

Then start each memory node with its local shard:
```bash
./scripts/start_memory_node.sh --index-file /path/to/index/shine_index_node1_of2.dat
```

Or let the compute-node initiator trigger startup loading on all memory nodes:
```bash
./scripts/start_compute_node.sh --load-index --index-prefix /path/to/index/shine_index
```

In both cases, the online cluster reuses the offline-built graph directly instead of rebuilding it through RDMA.

`--servers` can now be specified either as plain node names such as `cluster3` or as explicit `host:port`
endpoints such as `127.0.0.1:1235`. This allows running multiple memory nodes on the same machine as long as each
instance uses a distinct port.

Example: five memory nodes on one host with online load:

```bash
./scripts/build_offline_index.sh \
  --data-path /path/to/dataset-or-dir \
  --memory-nodes 5 \
  --threads 32 \
  --output-prefix /tmp/shine_index

./scripts/start_memory_node.sh --port 1234 --index-file /tmp/shine_index_node1_of5.dat
./scripts/start_memory_node.sh --port 1235 --index-file /tmp/shine_index_node2_of5.dat
./scripts/start_memory_node.sh --port 1236 --index-file /tmp/shine_index_node3_of5.dat
./scripts/start_memory_node.sh --port 1237 --index-file /tmp/shine_index_node4_of5.dat
./scripts/start_memory_node.sh --port 1238 --index-file /tmp/shine_index_node5_of5.dat

./scripts/start_compute_node.sh \
  --servers 127.0.0.1:1234 127.0.0.1:1235 127.0.0.1:1236 127.0.0.1:1237 127.0.0.1:1238 \
  --load-index \
  --index-prefix /tmp/shine_index \
  --dim 128 \
  --threads 16 \
  --coroutines 16 \
  --k 10 \
  --ef-search 32
```
