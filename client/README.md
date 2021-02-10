# Streaming Inference Client
Make streaming inference requests to the ensemble model exported in `~/export`, which should be sitting on a Google Kubernetes Engine (GKE) cluster somewhere. This will just report the latency and throughput achieved by the model; it won't do anything particularly interesting with the predictions the model produces (which is fine for now, since we're using a presumably dummy model and sending dummy data to it).

First build the conda environment (assuming you have conda installed):
```
conda env create -f conda/environment.yaml
```
Then activate it with
```
source activate gwe2e-client
```

Running is as simple as
```
./run.sh -u <url of GKE load balancer>
```

For example, at time of writing there should be such a load balancer available at

```
./run.sh -u 34.82.145.3:8001
```
If you're not running from the LIGO Data Grid (LDG) and so don't have access too real data, just use the `-d` flag to leverage dummy
data

```
./run.sh -u 34.82.145.3:8001 -d
```

If you plan on using multiple clients, you may want to specify a unique integer sequence id for each data stream using the `-s` flag (I have code to randomly generate them, but it's not working at the moment).

```
./run.sh -u 34.82.145.3:8001 -d -s 1001
```