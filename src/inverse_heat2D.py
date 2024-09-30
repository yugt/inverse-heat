import torch
from torchdiffeq import odeint
from motion2D import Motion2D


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InverseHeat2D:
    """
    Encapsulates the functionality for simulating and solving a 2D heat equation inverse problem.
    It estimates the conductivity of a medium based on temperature observations over time.
    """

    def __init__(self, J=16):
        """
        Initialize the spatial and temporal grid, initial conditions, and set up the basis functions.

        Args:
            J (int): Number of grid points along each spatial axis (resolution).
        """
        self.J = J
        self.x, self.y, self.t, self.xx, self.yy = self._initialize_grid(J)
        self.u0 = torch.cos(2 * torch.pi * self.xx) * torch.cos(2 * torch.pi * self.yy)
        self.f = lambda t: torch.sin(2 * torch.pi * t) * torch.ones_like(self.xx)
        self.conductivity = None  # Ground truth conductivity field
        self.c_guess = None       # Current guess of the conductivity during optimization
        self.sensor = None        # Sensor coordinates over time

    def _initialize_grid(self, J):
        """
        Initialize spatial and temporal grids.

        Args:
            J (int): Number of grid points along each spatial axis.

        Returns:
            Tuple[torch.Tensor]: Spatial (x, y) and temporal (t) grids, and meshgrid (xx, yy).
        """
        x = torch.linspace(0, 1, J + 1)[:-1]  # Discretized spatial coordinates in x
        y = torch.linspace(0, 1, J + 1)[:-1]  # Discretized spatial coordinates in y
        t = torch.linspace(0, 1, J * 4)  # Time grid for the simulation
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # 2D meshgrid for spatial domain
        return x, y, t, xx, yy

    def forward_Euler_1step(self, t, u):
        """
        Perform one step of the Euler finite difference scheme to approximate
        the time evolution of the heat equation.

        Note:
        - The parameter `a`, which represents the conductivity, cannot be passed
          directly as an argument to this function. This is because `torchdiffeq`
          requires the function signature for the ODE solver to be `f(t, u)`.
        - `self.c_guess` provides the value of `a` during forward integration.

        Args:
            t (torch.Tensor): Current time step (scalar).
            u (torch.Tensor): Current state variable (temperature distribution).

        Returns:
            torch.Tensor: Updated state variable after one Euler step.
        """
        a = self.c_guess
        # Shift `u` to approximate spatial derivatives in four directions
        u1 = torch.roll(u, dims=0, shifts=-1)   # Right neighbor
        u2 = torch.roll(u, dims=1, shifts=-1)   # Upward neighbor
        u3 = torch.roll(u, dims=0, shifts=1)    # Left neighbor
        u4 = torch.roll(u, dims=1, shifts=1)    # Downward neighbor

        # Shift `a` for conductivity in neighboring directions
        a1 = torch.roll(a, dims=0, shifts=1)    # Left conductivity
        a2 = torch.roll(a, dims=1, shifts=1)    # Downward conductivity

        # Finite difference approximation of heat equation
        return torch.stack([
            -u * (2 * a + a1 + a2),
            a * u1,
            a1 * u3,
            a * u2,
            a2 * u4
        ]).sum(dim=0) * a.shape[0] * a.shape[1] + self.f(t)

    def forward_PDE(self, a):
        """
        Solve the forward PDE for a given conductivity `a` over time using an ODE solver.
        This method integrates the heat equation using the `torchdiffeq` library.

        Args:
            a (torch.Tensor): Conductivity tensor.

        Returns:
            torch.Tensor: Temperature evolution over time.
        """
        # Store conductivity as `c_guess` for use in `forward_Euler_1step`
        self.c_guess = a
        return odeint(self.forward_Euler_1step, self.u0, self.t)

    def construct_Typewriter_basis(self, split=2):
        """
        Construct a block-wise basis (typewriter basis) for representing conductivity.

        Args:
            split (int): Number of blocks per axis in the typewriter basis.
        """
        assert(self.x.shape[0] % split == 0)  # Ensure the domain can be evenly split
        B = torch.ones((self.x.shape[0] // split, self.x.shape[0] // split))
        self.tbasis = torch.stack([
            torch.kron(self._generate_indicator_matrix(split, index=(i, j)), B)
            for i in range(split) for j in range(split)
        ])
        self.tbasis /= self.tbasis.norm(dim=(1, 2)).view(-1, 1, 1)

    def _generate_indicator_matrix(self, size, index=(0, 0)):
        """
        Generate an indicator matrix of the given size with a single non-zero entry.

        Args:
            size (int): Number of blocks per axis.
            index (tuple): Block index (i, j).

        Returns:
            torch.Tensor: A matrix with a single 1 at the specified block index.
        """
        A = torch.zeros((size, size))
        A[index] = 1
        return A

    def create_sensor(self, sensor_type, n=64):
        """
        Generate sensor positions for measuring the temperature field.
        Sensor positions are either moving or static over time.

        Args:
            sensor_type (str): Type of sensor placement, either 'moving' or 'static'.
            n (int): Number of sensors if static. Default is 64.
        """
        if sensor_type == 'moving':
            # Generate motion paths for moving sensors
            m = Motion2D(self.t.shape[0], self.x.shape[0])
            self.sensor = torch.stack([m.generate_motion(_) for _ in range(1, 17, 4)])
        elif sensor_type == 'static':
            # Initialize sensor positions for static sensors
            coordinates = []
            if n == 16:
                coordinates = [4, 12, 20, 28]
            elif n == 64:
                coordinates = [0, 4, 8, 12, 16, 20, 24, 28]
            else:
                raise ValueError(f"Unsupported number of static sensors: {n}")

            # Generate cartesian product for all coordinate pairs
            coordinates = torch.tensor(coordinates)
            # `self.sensor` is a tensor of shape [n, 2] representing the static sensor positions
            self.sensor = torch.cartesian_prod(coordinates, coordinates)
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")

    def get_moving_sensor_measurement(self, u):
        """
        Retrieve the temperature measurements from moving sensors.

        Args:
            u (torch.Tensor): Temperature evolution over time.

        Returns:
            torch.Tensor: Temperature measurements at sensor locations over time.
        """
        if self.sensor is None or not isinstance(self.sensor, torch.Tensor):
            raise ValueError("Sensor data is not initialized (None).")
        return u[self.sensor[:, 0, :], self.sensor[:, 1, :], self.sensor[:, 2, :]]


    def get_static_sensor_measurement(self, u):
        """
        Retrieve the temperature measurements from static sensors.

        Args:
            u (torch.Tensor): Temperature evolution over time.

        Returns:
            torch.Tensor: Temperature measurements at sensor locations over time.
        """
        # Ensure `self.sensor` is properly initialized
        if self.sensor is None or not isinstance(self.sensor, torch.Tensor):
            raise ValueError("Static sensors not initialized. Call `create_sensor` with `sensor_type='static'`.")

        # Collect temperature measurements at predefined static sensor positions
        return u[:, self.sensor[:, 0], self.sensor[:, 1]].T


def inverse_solver(a_truth, epoch, path='/tmp', frame_drawer=None, sensor_type='moving'):
    """
    Perform the inverse problem-solving loop with visualization.

    Args:
        a_truth (torch.Tensor): Ground truth conductivity tensor.
        epoch (int): Number of optimization steps.
        path (str): Directory path for saving output images.
        frame_drawer (Frame_Drawer): Object to handle visualization of frames.
        sensor_type (str): Type of sensors to use: "moving" or "static".

    Returns:
        tuple: The solver object (`ih2`) and a dictionary containing loss, error, and gradients.
    """

    # Preparation phase
    # Ensure the ground truth is square
    assert a_truth.shape[0] == a_truth.shape[1]
    J, _ = a_truth.shape

    # Initialize the solver for the forward PDE with grid size `J`
    ih2 = InverseHeat2D(J)
    ih2.conductivity = a_truth

    # Construct the typewriter basis (block basis) for the parameterization of the inverse problem
    ih2.construct_Typewriter_basis(split=J)

    # Solve the PDE forward in time to generate the temperature distribution `u`
    ih2.u = ih2.forward_PDE(a=ih2.conductivity)

    # Configure sensors based on type
    if sensor_type == 'moving':
        # Initialize moving sensors and extract their measurements
        ih2.create_sensor(sensor_type='moving')
        ih2.observe = ih2.get_moving_sensor_measurement(ih2.u)
    elif sensor_type == 'static':
        # Initialize static sensors and extract their measurements
        ih2.create_sensor(sensor_type='static', n=16)
        ih2.observe = ih2.get_static_sensor_measurement(ih2.u)
    else:
        raise ValueError(f"Unsupported sensor type: {sensor_type}")

    # Initialize the `frame_drawer` for visualizing the results
    frame_drawer.t, frame_drawer.x, frame_drawer.y = ih2.t, ih2.x, ih2.y
    frame_drawer.path = path
    frame_drawer.ground_truth(ih2.conductivity)
    frame_drawer.temperature_sensor(ih2.u, ih2.sensor, ih2.observe)

    # Inverse solver initialization
    l2 = torch.nn.MSELoss()
    noise = torch.randn(ih2.observe.shape) * 0.0  # Add noise if necessary (currently 0)
    denominator = l2(ih2.conductivity, torch.zeros_like(ih2.conductivity))

    # Initialize `atm` as the log of a uniform guess for the conductivity field
    atm = torch.log(torch.ones(J * J) / 120)

    # Prepare storage for optimization metrics (loss, error, gradients, etc.)
    df = {
        'loss': torch.full((epoch,), float('nan')),
        'error': torch.full((epoch,), float('nan')),
        'tbasis': [],  # Stores parameter values at each epoch
        'grad': []     # Stores gradients at each epoch
    }

    # Uncomment for optional gradient descent optimizer initialization
    '''
    gradescent = lambda params: torch.optim.SGD([params], lr=1e4,
            momentum=0.1, dampening=0.1, weight_decay=0) ## steepest descent
    opt = gradescent(atm)
    '''

    # Optimization loop over the specified number of epochs
    for _ in range(epoch):
        # opt.zero_grad()  # Uncomment for optional optimizer initialization
        # Set gradients to be computed for `atm`
        atm.requires_grad = True

        # Construct the conductivity guess using the typewriter basis
        a_guess = torch.exp((ih2.tbasis * atm.view(-1, 1, 1)).sum(dim=0))

        # Simulate temperature measurements using the guessed conductivity
        if sensor_type == 'moving':
            # Use moving sensors to get measurements
            calculate = ih2.get_moving_sensor_measurement(ih2.forward_PDE(a=a_guess))
        elif sensor_type == 'static':
            # Use static sensors (16) to get measurements
            calculate = ih2.get_static_sensor_measurement(ih2.forward_PDE(a=a_guess), n=16)
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")

        # Compute the loss as the difference between observed and simulated measurements
        loss = l2(ih2.observe + noise, calculate)

        # Perform backpropagation to compute gradients w.r.t `atm`
        loss.backward()

        # Store results of the current epoch
        df['tbasis'].append(atm.detach().numpy())  # Store parameters
        df['grad'].append(atm.grad.detach().numpy())  # Store gradients
        df['loss'][_] = loss.detach()  # Store loss
        df['error'][_] = l2(ih2.conductivity, a_guess.detach()) / denominator  # Store relative error

        # Visualize the current state of the inverse solution using `frame_drawer`
        frame_drawer.inverse_frame(
            a_guess.detach().numpy(),
            calculate.detach().numpy(),
            atm.grad.detach().reshape((J, J)).numpy(),
            df['loss'].numpy(),
            df['error'].numpy()
        ).write_image(f'{path}/epoch-{_:03d}.png')  # Save frame as an image

        # Manual gradient descent update step (update `atm`)
        atm = atm.detach() - atm.grad.detach() * 1e4

        # Uncomment to use the optional optimizer step instead
        # opt.zero_grad()
        # opt.step()

    return ih2, df