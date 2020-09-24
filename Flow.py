import torch
import torch.nn.functional as F
import skimage


# input must be 4-dim
class Flow:
    def __init__(self, field=None, sz=None):
        if sz is None:
            self.field = field
            self.base_field = None
            if field is not None:
                self.create_field(field)
        else:
            theta = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().unsqueeze(0)
            self.base_field = F.affine_grid(theta, [1, 1, sz[-2], sz[-1]])
            field = F.affine_grid(field, sz) - self.base_field.to(field.device)
            if field.shape[-1] == 2:
                self.field = field.permute([0, 3, 1, 2])

    # return a torch
    def draw_flow(self, alpha=1):
        assert self.field is not None, "please create a field"
        field = self.field.detach()
        pi = 3.141592653
        deltay = field[:, 1:]
        deltax = field[:, :1]
        h = (torch.atan(deltay / (deltax + 1e-7)) / (2 * pi) + 0.25 + (deltax >= 0) * 0.5).cpu()
        s = alpha * ((deltax ** 2 + deltay ** 2) ** 0.5 / 2).cpu()
        v = torch.ones(h.shape)
        color_field = torch.cat([h, s, v], dim=1).permute([0, 2, 3, 1])
        l = []
        for f in color_field.numpy():
            l.append(skimage.color.hsv2rgb(f))
        color_field = torch.tensor(l).permute([0, 3, 1, 2])
        return color_field

    def sample(self, src):
        assert self.field is not None, "please create a field"
        grid = self.field.permute([0, 2, 3, 1]) + self.base_field.to(self.field.device)
        warped = F.grid_sample(src, grid)
        return warped

    def create_field(self, field, plus=None):
        if plus is None or self.field is None:
            self.field = field
        else:
            assert self.field.shape == field.shape, "Size of field is not equal"
            self.field = self.field + field
        if field.shape[-1] == 2:
            self.field = field.permute([0, 3, 1, 2])
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().unsqueeze(0)
        self.base_field = F.affine_grid(theta, [1, 1, field.shape[-2], field.shape[-1]])

    # 50-250
    def create_deform(self, pt, mv, sz=None, szm=None):
        if szm is None:
            szm = [512, 512]
        if sz is None:
            sz = [1, 2, 512, 512]
        d_field = torch.zeros(sz)
        w, h = sz[-2:]
        wm, hm = szm[-2:]
        wm = wm // 2
        hm = hm // 2
        center_x = pt[0]
        center_y = pt[1]
        # 横向移动
        for x in range(wm):
            delta_x = mv[0]
            for y in range(hm):
                alpha = (1 - (x / wm) ** 2 - (y / hm) ** 2) ** 2 * (((x / wm) ** 2 + (y / hm) ** 2) < 1)
                if center_x - x > 0 and center_y - y > 0:
                    d_field[0][0][center_x - x][center_y - y] = delta_x * alpha
                if center_x + x < sz[-2] and center_y - y > 0:
                    d_field[0][0][center_x + x][center_y - y] = delta_x * alpha
                if center_x - x > 0 and center_y + y < sz[-1]:
                    d_field[0][0][center_x - x][center_y + y] = delta_x * alpha
                if center_x + x < sz[-2] and center_y + y < sz[-1]:
                    d_field[0][0][center_x + x][center_y + y] = delta_x * alpha
        for y in range(hm):
            delta_y = mv[1]
            for x in range(wm):
                alpha = (1 - (x / wm) ** 2 - (y / hm) ** 2) ** 2 * (((x / wm) ** 2 + (y / hm) ** 2) < 1)
                if center_x - x > 0 and center_y - y > 0:
                    d_field[0][1][center_x - x][center_y - y] = delta_y * alpha
                if center_x + x < sz[-2] and center_y - y > 0:
                    d_field[0][1][center_x + x][center_y - y] = delta_y * alpha
                if center_x - x > 0 and center_y + y < sz[-1]:
                    d_field[0][1][center_x - x][center_y + y] = delta_y * alpha
                if center_x + x < sz[-2] and center_y + y < sz[-1]:
                    d_field[0][1][center_x + x][center_y + y] = delta_y * alpha
        if abs(mv[0]) + abs(mv[1]) > 2:
            d_field[0][0] /= w
            d_field[0][1] /= h
        self.create_field(d_field, "+")


# input: 3-dim torch, output:3-dim torch
def random_deform(img, pn=2, NeedFlow=False):
    f = Flow()
    for i in range(pn):
        pt = torch.randint(100, 412, [2]).numpy()
        mv = torch.randint(-128, 128, [2]).numpy()
        f.create_deform(pt, mv, szm=[256, 256])
        img = f.sample(img.unsqueeze(0)).squeeze(0)
    if NeedFlow:
        return img, f.field.squeeze(0)
    else:
        return img


__all__ = {"Flow", "random_deform"}
# a = torch.arange(512).reshape([1, 1, 1, 512])
# b = a.reshape([1, 1, 512, 1])
# x = a.repeat([1, 1, 512, 1]).float() / 255 - 1
# y = b.repeat([1, 1, 1, 512]).float() / 255 - 1
# fi = Flow(torch.cat([x, y], dim=1))
# cf = fi.draw_flow().squeeze(0)
# tr = transforms.ToPILImage()
# cfile = tr(cf)
# cfile.save("cf.bmp")
