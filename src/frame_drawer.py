import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np
import time


class Frame_Drawer:
    def __init__(self, t, x, y, epoch, path="tmp"):
        self.x = x
        self.y = y
        self.t = t
        self.epoch = np.arange(epoch)
        self.path = path

        self._initialize_figure()

    def _initialize_figure(self):
        """Set up the initial structure of the figure with subplots and layout."""
        self.fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                r'$\text{Ground truth } a$',
                r'$\text{Temperature } u(t,\cdot) \text{ and sensors } s_k(t)$',
                r'$\text{Sensor measurements } u(t, s_k(t))$',
                r'$\text{Current guess } a$',
                r'$\text{Gradient of loss } \nabla_\theta L$',
                r'$\text{Residual } u(t, s_k(t);a_\text{truth})-u(t, s_k(t);a_\text{guess})$'
            ),
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )

        self.fig.update_layout(
            width=1280, height=720,
            margin=dict(l=30, r=30, t=20, b=20)
        )

    def ground_truth(self, a):
        """Add the ground truth heatmap to the figure."""
        self._add_heatmap(a, row=1, col=1, colorscale='plasma', zmin=0.005, zmax=0.025,
                          colorbar_x=0.265, colorbar_y=0.775, name='a_truth')

    def _add_heatmap(self, z_data, row, col, colorscale, zmin, zmax, colorbar_x, colorbar_y,
                    tickvals=None, name=None):
        """Helper method to add a heatmap to the specified subplot with control over xaxis and yaxis."""
        # Add the heatmap trace
        self.fig.add_trace(
            go.Heatmap(
                x=self.x, y=self.y, z=z_data, zmin=zmin, zmax=zmax,
                colorscale=colorscale,
                colorbar=dict(
                    showticklabels=True, thickness=20, x=colorbar_x, y=colorbar_y,
                    len=0.48,
                    tickvals=tickvals
                ),
                showscale=True,
                name=name
            ),
            row=row, col=col
        )

    def _snapshot(self, frame_idx, name, colorbar_x, colorbar_y):
        """Generate a contour plot for a given frame index with customizable colorbar position."""
        return go.Contour(
            x=self.x, y=self.y, z=self.u[frame_idx],
            colorscale='jet',
            zmin=-1, zmax=1,
            colorbar=dict(
                showticklabels=True, thickness=20, x=colorbar_x, y=colorbar_y,
                len=1, tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
            ),
            contours=dict(coloring='heatmap', showlines=True, start=-1, end=1, size=0.25),
            line=dict(width=1, color='black'),
            showscale=True,
            name=name
        )

    def _add_sensor_summary(self, fig, row=None, col=None):
        """
        Add sensor measurement traces to the specified figure.

        Args:
            fig (plotly.graph_objects.Figure): The figure to which traces are added.
            row (int, optional): Row of subplot for adding traces (used only if `fig` is a subplot).
            col (int, optional): Column of subplot for adding traces (used only if `fig` is a subplot).
        """
        # Add all sensor traces
        for i in range(self.observe.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=self.observe[i],
                    mode='lines',
                    line=dict(width=1),
                    name=f'{i+1}',
                    showlegend=self.observe.shape[0] <= 16
                ),
                row=row, col=col
            )

        # Set x-axis and y-axis titles
        fig.update_xaxes(title_text=r'$t$', title_standoff=0, row=row, col=col)
        fig.update_yaxes(title_text=r'$u(t, s_k(t))$', title_standoff=0, range=[-1, 1], row=row, col=col)

        # Adjust layout and legend position
        fig.update_layout(
            width=1280,
            height=300,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(
                x=1, y=0,  # Position legend at the right bottom
                xanchor='right', yanchor='bottom',
                orientation='h',
                traceorder='normal'
            )
        )


    def _summary(self):
        """
        Create a summary plot: if `self.observe.shape[0] == 4`, create a 1x4 subplot;
        otherwise, create a figure with only sensor measurements.
        """
        if self.observe.shape[0] == 4:
            # Create the 1x4 subplot
            fig = make_subplots(
                rows=1, cols=4,
                subplot_titles=[
                    r"$u(t=0,\cdot)$",
                    r"$u(t=0.5,\cdot)$",
                    r"$u(t=1,\cdot)$",
                    r'$\text{Sensor measurements}$'
                ],
                horizontal_spacing=0.1
            )

            # Add contour plots for t=0, t=0.5, and t=1
            fig.add_trace(self._snapshot(0, "t=0", 0.17, 0.5), row=1, col=1)
            fig.add_trace(self._snapshot(self.u.shape[0] // 2, "t=0.5", 0.45, 0.5), row=1, col=2)
            fig.add_trace(self._snapshot(self.u.shape[0] - 1, "t=1", 0.72, 0.5), row=1, col=3)

            # Add sensor traces to the fourth subplot
            self._add_sensor_summary(fig, row=1, col=4)

        else:
            # Create a 1x2 subplot where the second column is three times wider
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[1, 4],  # Set the second column to be 3 times wider than the first
                subplot_titles=[
                    r'$u(t=0,\cdot) \text{ and static sensor positions}$',  # Adjust title for the plots
                    r'$\text{static sensor measurements } u(t,s_k)$'
                ],
                horizontal_spacing=0.1  # Adjust spacing between subplots
            )

            fig.add_trace(self._contour_plot(0, 0.18, 0.5, 1), row=1, col=1)
            fig.add_trace(self._static_sensor_scatter(), row=1, col=1)
            # Add sensor traces to the second subplot (row=1, col=2)
            self._add_sensor_summary(fig, row=1, col=2)

        time.sleep(1)
        return fig


    def temperature_sensor(self, u, sensor, observe):
        """Visualize temperature and sensor data over time."""
        assert u.shape[0] == self.t.shape[0]
        assert observe.shape[-1] == self.t.shape[0]

        self.u = u
        self.sensor = sensor
        self.observe = observe
        self.sensor_color = [ pc.qualitative.Plotly[_ % len(pc.qualitative.Plotly)]
                                for _ in range(self.sensor.shape[0]) ]

        for frame_idx in range(u.shape[0]):
            self._forward_frame(frame_idx).write_image(f'{self.path}/forward-{frame_idx:03d}.png')

        self._summary().write_image(f'{self.path}/forward.pdf', format='pdf')

        self._initialize_inverse()

    def _initialize_inverse(self):
        """Initialize the inverse figure layout after the forward frame drawing."""
        temp_fig = self._create_subplots()
        for trace in self.fig.data:
            temp_fig.add_trace(trace)
        self.fig = temp_fig

    def _create_subplots(self):
        """Create a subplot layout for the inverse frame."""
        return make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                r'$\text{Ground truth } a$',
                r'$\log_{10}L(\theta) \text{ and } \log_{2}{\|a_\text{truth} - a_\text{guess}\|}/{\|a_\text{truth}\|}$',
                r'$u(t, s_k(t);a_\text{truth}) \text{ and } u(t, s_k(t);a_\text{guess})$',
                r'$\text{Current guess } a_\text{guess}(\cdot;\theta)=\exp(\theta)$',
                r'$\text{Gradient of loss } \nabla_\theta L$',
                r'$\text{Residual } u(t, s_k(t);a_\text{truth})-u(t, s_k(t);a_\text{guess})$'
            ),
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        ).update_layout(
            width=1280, height=720,
            margin=dict(l=30, r=30, t=20, b=20)
        )

    def _remove_axes(self, axes_to_remove):
        """Remove traces from specific axes."""
        self.fig.data = [
            trace for trace in self.fig.data
            if not any((trace.xaxis == x and trace.yaxis == y) for x, y in axes_to_remove)
        ]

    def _forward_frame(self, frame_idx):
        """Add contour and sensor data to the figure for the current frame."""
        self._remove_axes([('x2', 'y2'), ('x3', 'y3')])

        self.fig.add_trace(self._contour_plot(frame_idx), row=1, col=2)
        self._add_sensor_measurements(frame_idx)
        if self.observe.shape[0] == 4:
            self._add_sensor_scatter(frame_idx)
        else:
            self._add_sensor_scatter(None)

        self.fig.update_layout(
            yaxis3=dict(range=[-1, 1]),
            xaxis3=dict(title=r'$t$'),
            legend=dict(
                x=1, y=0.55,
                xanchor='right', yanchor='bottom',
                orientation='v',
                traceorder='normal'
            )
        )

        return self.fig

    def _contour_plot(self, frame_idx, colorbar_x=0.63, colorbar_y=0.775, colorbar_len=0.48):
        """
        Return a contour plot for the given frame with adjustable colorbar position and length.

        Args:
            frame_idx (int): The index of the frame to plot.
            colorbar_x (float): Horizontal position of the colorbar.
            colorbar_y (float): Vertical position of the colorbar.
            colorbar_len (float): Length of the colorbar.

        Returns:
            go.Contour: A contour plot object.
        """
        return go.Contour(
            x=self.x, y=self.y, z=self.u[frame_idx],
            colorscale='jet',
            zmin=-1, zmax=1,
            colorbar=dict(
                showticklabels=True,
                thickness=20,
                x=colorbar_x, y=colorbar_y,  # Use provided positions
                len=colorbar_len,  # Use provided length
                tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
            ),
            contours=dict(coloring='heatmap', showlines=True, start=-1, end=1, size=0.25),
            line=dict(width=1, color='black'),
            showscale=True
        )



    def _static_sensor_scatter(self):
        return go.Scatter(
                    x=self.x[self.sensor[:, 0]],
                    y=self.y[self.sensor[:, 1]],
                    mode='markers',
                    marker=dict(size=5, color=self.sensor_color),
                    name='sensors',
                    legendgroup='row1col2',
                    showlegend=False
                )


    def _add_sensor_scatter(self, frame_idx):
        """Add scatter plot for sensors."""
        if frame_idx is None:
            self.fig.add_trace(
                self._static_sensor_scatter(),
                row=1, col=2)
        else:
            self.fig.add_trace(
                go.Scatter(
                    x=self.x[self.sensor[:, 1, frame_idx]],
                    y=self.y[self.sensor[:, 2, frame_idx]],
                    mode='markers',
                    marker=dict(size=5, color=self.sensor_color),
                    name='sensors',
                    legendgroup='row1col2',
                    showlegend=False
                ),
                row=1, col=2
            )

    def _add_sensor_measurements(self, frame_idx):
        """Add sensor measurements as scatter plots in row=1, col=3."""
        show_legend = True
        name = lambda i: f's{i+1:02d}'
        if self.observe.shape[0] <= 4:
            name = lambda i: f'sensor {i+1}'
        elif self.observe.shape[0] < 9:
            name = lambda i: f's{i+1}'
        elif self.observe.shape[0] <= 16:
            pass
        else:
            show_legend = False
        for i in range(self.observe.shape[0]):
            y_full_length = np.full(self.t.shape[0], np.nan)
            y_full_length[:frame_idx] = self.observe[i, :frame_idx]

            self.fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=y_full_length,
                    mode='lines',
                    line=dict(width=1),
                    name=name(i),
                    legendgroup='row1col3',
                    showlegend=show_legend
                ),
                row=1, col=3
            )

    def inverse_frame(self, a, c, g, loss, error):
        """Handle inverse frame visualization."""
        self._remove_axes([('x2', 'y2'), ('x3', 'y3'), ('x4', 'y4'), ('x5', 'y5'), ('x6', 'y6')])
        self._remove_loss_error_annotations()

        self._plot_inverse_heatmaps(a, g)
        self._plot_loss_error_traces(loss, error)
        self._add_loss_error_annotations(loss, error)
        self._plot_sensor_data(c)

        # Update layout: Set x-axis and y-axes for row=1, col=2
        self.fig.update_layout(
            xaxis2=dict(
                title='epoch+1',
                tickmode='array',
                tickvals=np.log10([1, 10, 100, 1+self.epoch[-1]]),  # Log10 positions of the ticks including last epoch
                ticktext=['1', '10', '100', f'{1+self.epoch[-1]}'], # Corresponding tick labels
                type='linear'  # Keep the axis in log scale
            ),
            xaxis3=dict(title=r'$t$'),
            xaxis6=dict(title=r'$t$'),
            legend=dict(
                x=1, y=0,
                xanchor='right', yanchor='bottom',
                orientation='h' if self.observe.shape[0] == 4 else 'v',
                traceorder='normal'
            )
        )

        return self.fig

    def _remove_loss_error_annotations(self):
        """Remove existing 'Loss' and 'Error' annotations."""
        self.fig.layout.annotations = [
            annotation for annotation in self.fig.layout.annotations
            if annotation['text'] not in ['Loss', 'Error']
        ]

    def _plot_loss_error_traces(self, loss, error):
        """Add loss and error traces to the figure."""
        self.fig.add_trace(
            go.Scatter(
                x=np.log10(1+self.epoch),
                y=np.log10(loss),
                mode='lines',
                line=dict(width=2),
                name='loss',
                legendgroup='loss_error',
                xaxis='x2',
                yaxis='y2',
                showlegend=False
            ),
            row=1, col=2
        )

        self.fig.add_trace(
            go.Scatter(
                x=np.log10(1+self.epoch),
                y=np.log2(error),
                mode='lines',
                line=dict(width=2),
                name='error',
                legendgroup='loss_error',
                showlegend=False,
                xaxis='x2',
                yaxis='y2'
            ),
            row=1, col=2
        )


    def _add_loss_error_annotations(self, loss, error):
        """Add annotations for 'Loss' and 'Error'."""
        epoch = np.where(~np.isnan(loss))[0][-1]

        self.fig.add_annotation(
            text='Loss',
            x=np.log10(1+self.epoch[epoch]),
            y=np.log10(loss[epoch]),
            xref='x2', yref='y2',
            showarrow=True, arrowhead=2, ax=0, ay=-30
        )

        self.fig.add_annotation(
            text='Error',
            x=np.log10(1+self.epoch[epoch]),
            y=np.log2(error[epoch]),
            xref='x2', yref='y2',
            showarrow=True, arrowhead=2, ax=0, ay=30
        )

    def _plot_inverse_heatmaps(self, a, g):
        """Plot heatmaps for the current guess and gradient."""
        self._add_heatmap(a, row=2, col=1, colorscale='plasma', zmin=0.005, zmax=0.025,
                          colorbar_x=0.265, colorbar_y=0.225, name='a_guess')
        self._add_heatmap(g, row=2, col=2, colorscale='piyg', zmin=None, zmax=None,
                           colorbar_x=0.63, colorbar_y=0.225, name='gradient')

    def _plot_sensor_data(self, c):
        """Plot sensor data and residuals for the inverse frame."""
        for i in range(self.observe.shape[0]):
            self.fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=self.observe[i],
                    mode='lines',
                    line=dict(width=1),
                    name=f's{i+1}',
                    legendgroup=f'sensor_{i+1}',
                    showlegend=False
                ),
                row=1, col=3
            )
            self.fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=c[i],
                    mode='lines',
                    line=dict(width=1),
                    name=f's{i+1}',
                    legendgroup=f'sensor_{i+1}',
                    showlegend=False
                ),
                row=1, col=3
            )

        for i in range(c.shape[0]):
            self.fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=self.observe[i] - c[i],
                    mode='lines',
                    line=dict(width=1),
                    name=f's{i+1}',
                    legendgroup=f'sensor_{i+1}',
                    showlegend=(self.observe.shape[0] <= 10)
                ),
                row=2, col=3
            )
