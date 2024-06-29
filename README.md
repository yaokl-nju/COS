Codes for "Context Sketching for Memory-efficient Graph Representation Learning" (ICDM 2023).

README for codes on ogbg-molhiv and ogbg-molpcba

Depencies:
    1. torch version 1.8.0+cu11.1
    2. torch_geometric version 1.6.3
    3. torch_sparse version 0.6.10
    4. torch_scatter version 2.0.7
    5. torch_cluster version 1.5.9
    6. ogb version 1.3.1
    7. Others that are missed. Readers can install them by yourselves.

Run following command to generate results:
	python main_ogbg.py


### Generate results for baselines.
	reset config-dataset.yaml:
		local_method: false

### Generate results for COS.
	reset config-dataset.yaml:
		local_method: true

## Citation
<pre>
@inproceedings{DBLP:conf/icdm/YaoL23,
  author       = {Kai{-}Lang Yao and
                  Wu{-}Jun Li},
  title        = {Context Sketching for Memory-efficient Graph Representation Learning},
  booktitle    = {{IEEE} International Conference on Data Mining},
  year         = {2023},

}
</pre>
