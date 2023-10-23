import torch

window_size=(3, 2, 2)

coords_d = torch.arange(window_size[0])
coords_h = torch.arange(window_size[1])
coords_w = torch.arange(window_size[2])
coords = torch.stack(torch.meshgrid(
    coords_d,
    coords_h,
    coords_w,
))  # 3, Wd, Wh, Ww

coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

relative_coords = relative_coords.permute(1, 2, 0).contiguous()

# shift to start from 0
relative_coords[:, :, 0] += window_size[0] - 1
relative_coords[:, :, 1] += window_size[1] - 1
relative_coords[:, :, 2] += window_size[2] - 1

relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * \
                                    (2 * window_size[2] - 1)
relative_coords[:, :, 1] *= (2 * window_size[2] - 1)

relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww


print('---------------')
relative_position_index = relative_position_index.view(window_size[0],window_size[1]*window_size[2],window_size[0],window_size[1]*window_size[2]).permute(0,2,1,3).reshape(window_size[0]*window_size[0],window_size[1]*window_size[2], window_size[1]*window_size[2])
print(relative_position_index)
relative_position_index = relative_position_index[::window_size[0],:,:]
print('---------------')
print(relative_position_index)
print(relative_position_index.shape)

relative_position_bias_table = \
            torch.rand((2 * window_size[0] - 1) * (2 * window_size[1] - 1) *
                        (2 * window_size[2] - 1), 2)

relative_position_bias = relative_position_bias_table[relative_position_index[:].reshape(-1),:]
# .reshape(frame_len, N, N, -1)  # 8frames ,Wd*Wh*Ww,Wd*Wh*Ww,nH

print(relative_position_bias.shape)



