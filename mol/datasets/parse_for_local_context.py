import torch
from tqdm import tqdm

def parse_for_local_context(g_list, cs):
    print("parse local context for each node begins......")
    num_atom_type, num_bond_type = g_list['num_atom_type'], g_list['num_bond_type']
    edges, bond_type, indptr_e, indptr_x, indptr_n = g_list['edges'], g_list['bond_type'], g_list['indptr_e'], g_list['indptr_x'], g_list['indptr_n']
    node_color_map_pre, num_new_color = g_list['atom_type'], num_atom_type

    mapping, record = {}, {}
    for csi in range(cs):
        node_color_map_cur, num_cur_color = node_color_map_pre, num_new_color
        node_color_map_pre, num_new_color = torch.zeros(node_color_map_cur.size(0), dtype=torch.long) - 1, 0

        cs_node, cs_count, cs_color = {}, {}, {}
        for i in range(num_cur_color):
            cs_node[i] = {}
            cs_count[i] = {}

        for g in tqdm(range(indptr_x.size(0) - 1), desc='Extract local context for each node.'):
            # resort edges
            edges_g = edges[:, indptr_e[g]: indptr_e[g + 1]]
            bond_type_g = bond_type[indptr_e[g]: indptr_e[g + 1]]
            perm = torch.argsort(
                edges_g[0] * (num_cur_color * num_bond_type) + (node_color_map_cur[edges_g[1] + indptr_x[g]] * num_bond_type + bond_type_g)
            )
            edges_g = edges_g[:, perm] + indptr_x[g]
            bond_type_g = bond_type_g[perm]
            indptr_n_g = indptr_n[g]

            for i in range(indptr_n_g.size(0) - 1):
                global_i = i + indptr_x[g]
                src_node_color = node_color_map_cur[global_i].item()
                neighs = edges_g[1, indptr_n_g[i]: indptr_n_g[i + 1]]
                neighs_atom = node_color_map_cur[neighs]
                neighs_bond = bond_type_g[indptr_n_g[i]: indptr_n_g[i + 1]]
                enc_ids = tuple((neighs_atom * num_bond_type + neighs_bond).numpy())
                if enc_ids not in cs_node[src_node_color]:
                    cs_node[src_node_color][enc_ids] = [global_i]
                    key = (src_node_color, *enc_ids)
                    new_id = len(cs_color)
                    cs_color[key] = new_id
                    node_color_map_pre[global_i] = new_id
                else:
                    cs_node[src_node_color][enc_ids].append(global_i)
                    key = (src_node_color, *enc_ids)
                    node_color_map_pre[global_i] = cs_color[key]

        num_new_color = len(cs_color)

        shift = node_color_map_cur.size(0)
        mapping_c = torch.zeros(shift, dtype=torch.long) - 1
        record_c = torch.zeros(shift, dtype=torch.bool)
        for aid in range(num_cur_color):
            context_i = cs_node[aid]
            if len(context_i) > 0:
                for key in context_i.keys():
                    if len(context_i[key]) > 1:
                        for nid_o in context_i[key]:
                            mapping_c[nid_o] = shift
                            record_c[nid_o] = True
                        shift += 1

        mapping[csi] = mapping_c
        record[csi] = record_c

    return mapping, record