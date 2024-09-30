import torch
import numpy as np

class Motion2D:
    def __init__(self, Nt, Nx, Ny=None) -> None:
        if Ny is None:
            Ny = Nx
        assert(Nt % 4 == 0)
        self.t = torch.linspace(0, 1, Nt)
        self.x = torch.linspace(0, 1, Nx+1)[:-1]  # Ensure periodicity in [0,1)
        self.y = torch.linspace(0, 1, Ny+1)[:-1]  # Ensure periodicity in [0,1)

    def generate_motion(self, n=1):
        '''
        stack n such motion paths along the first dimension:
        tensor([[[ 0,  1,  2, ..., 15],  # Time indices for sensor 1
         [ 5,  5,  9, ...,  5],  # x-coordinates for sensor 1
         [ 5,  9,  9, ...,  5]],  # y-coordinates for sensor 1

        [[ 0,  1,  2, ..., 15],  # Time indices for sensor 2
         [10, 10, 14, ..., 10],  # x-coordinates for sensor 2
         [10, 14, 14, ..., 10]],  # y-coordinates for sensor 2

         ...

        [[ 0,  1,  2, ..., 15],  # Time indices for sensor n
         [20, 20, 24, ..., 20],  # x-coordinates for sensor n
         [20, 24, 24, ..., 20]]])  # y-coordinates for sensor n
        '''
        if n in [1,2,3,4]:
            _ = self.t.shape[0]//4
            _ = [
                torch.linspace(0.1, 0.9, _),
                torch.linspace(0.9, 0.9, _),
                torch.linspace(0.9, 0.1, _),
                torch.linspace(0.1, 0.1, _)]
        elif n in [5,6,7,8]:
            _ = self.t.shape[0]//8
            _ = [
                torch.linspace(0.2, 0.8, _),
                torch.linspace(0.8, 0.8, _),
                torch.linspace(0.8, 0.2, _),
                torch.linspace(0.2, 0.2, _)]*2
        elif n in [9, 10, 11, 12]:
            _ = self.t.shape[0]//16
            _ = [
                torch.linspace(0.3, 0.7, _),
                torch.linspace(0.7, 0.7, _),
                torch.linspace(0.7, 0.3, _),
                torch.linspace(0.3, 0.3, _)]*4
        else:
            _ = self.t.shape[0]//32
            _ = [
                torch.linspace(0.4, 0.6, _),
                torch.linspace(0.6, 0.6, _),
                torch.linspace(0.6, 0.4, _),
                torch.linspace(0.4, 0.4, _)]*8
        if n%4 == 1:
            _ = torch.stack([torch.cat(_[-1:] + _[:-1]), torch.cat(_[-2:] + _[:-2])])
        elif n%4 == 2:
            _ = torch.stack([torch.cat(_[-2:] + _[:-2]), torch.cat(_[-3:] + _[:-3])])
        elif n%4 == 3:
            _ = torch.stack([torch.cat(_[-3:] + _[:-3]), torch.cat(_)])
        else:
            _ = torch.stack([torch.cat(_), torch.cat(_[-1:] + _[:-1])])
        return torch.cat([torch.arange(end=self.t.shape[0]).unsqueeze(0),
                    torch.round(_*self.x.shape[0])]).to(torch.int)

    def generate_spiral(self):
        sx = np.vectorize(self._spiral_x)
        sy = np.vectorize(self._spiral_y)
        x = np.int32(sx(self.t.numpy())*self.x.shape[0])
        y = np.int32(sy(self.t.numpy())*self.y.shape[0])
        return torch.stack([
            torch.arange(end=self.t.shape[0]),
            torch.from_numpy(x),
            torch.from_numpy(y)])

    def _spiral_x(self, t):
            if t<0.1:
                return 0.1
            elif t<0.2:
                return 0.1 + 8 * (t-0.1)
            elif t<0.3:
                return 0.9
            elif t<0.3875:
                return 0.9 - 8 * (t-0.3)
            elif t<0.475:
                return 0.2
            elif t<0.55:
                return 0.2 + 8 * (t-0.475)
            elif t<0.625:
                return 0.8
            elif t<0.6875:
                return 0.8 - 8 * (t-0.625)
            elif t<0.75:
                return 0.3
            elif t<0.8:
                return 0.3 + 8 * (t-0.75)
            elif t<0.85:
                return 0.7
            elif t<0.8875:
                return 0.7 - 8 * (t-0.85)
            elif t<0.925:
                return 0.4
            elif t<0.95:
                return 0.4 + 8 * (t-0.925)
            elif t<0.975:
                return 0.6
            elif t<0.9875:
                return 0.6 - 8 * (t-0.975)
            elif  t<=1:
                return 0.5
            else:
                raise ValueError('t must be in [0,1]')

    def _spiral_y(self, t):
            if t<0.1:
                return 0.9 - 8 * t
            elif t<0.2:
                return 0.1
            elif t<0.3:
                return 0.1 + 8 * (t-0.2)
            elif t<0.3875:
                return 0.9
            elif t<0.475:
                return 0.9 - 8 * (t-0.3875)
            elif t<0.55:
                return 0.2
            elif t<0.625:
                return 0.2 + 8 * (t-0.55)
            elif t<0.6875:
                return 0.8
            elif t<0.75:
                return 0.8 - 8 * (t-0.6875)
            elif t<0.8:
                return 0.3
            elif t<0.85:
                return 0.3 + 8 * (t-0.8)
            elif t<0.8875:
                return 0.7
            elif t<0.925:
                return 0.7 - 8 * (t-0.8875)
            elif t<0.95:
                return 0.4
            elif t<0.975:
                return 0.4 + 8 * (t-0.95)
            elif t<0.9875:
                return 0.6
            elif  t<=1:
                return 0.6 - 8 * (t-0.9875)
            else:
                raise ValueError('t must be in [0,1]')



if __name__ == "__main__":
    motion2d = Motion2D(320, 32, 32)
    print(motion2d.generate_motion(n=1))