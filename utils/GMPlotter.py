import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn import mixture
from scipy.interpolate import make_interp_spline
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses
plt.rcParams.update({'font.size': 18})

color_iter = itertools.cycle(["deepskyblue", "cornflowerblue", "darkturquoise",
                               "lightblue", "mediumturquoise", "powderblue", "skyblue"])

class GaussianMixturePlotter:
    def __init__(self, robotName, actionName, angleId, n_components=10, babbled_points=30, covariance_type="full", max_iter=150, random_state=2):
        """
        Initialize the GMPlotter object.
        This class is used to fit, interpolate and plot the results of the Gaussian Mixture Models and Gaussian Mixture Regression.

        Args:
            robotName (str): The name of the robot.
            actionName (str): The name of the action.
            angleId (int): The ID of the angle.
            n_components (int, optional): The number of mixture components. Defaults to 10.
            babbled_points (int, optional): The number of babbled points. Defaults to 30.
            covariance_type (str, optional): The type of covariance. Defaults to "full".
            max_iter (int, optional): The maximum number of iterations. Defaults to 150.
            random_state (int, optional): The random state. Defaults to 2.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.robotName = robotName
        self.actionName = actionName
        self.angleId = angleId
        self.babbled_points = babbled_points

    def fit_gaussian_mixture(self, X):
        """
        Fits a Gaussian Mixture Model to the given data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        - gmm: GaussianMixture object
            The fitted Gaussian Mixture Model.
        """
        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state
        ).fit(X)
        return gmm

    def save_gmr(self, x_smooth_range, y_smooth_range):
        """
        Save the Gaussian Mixture Regression (GMR) data to a CSV file.

        Args:
            x_smooth_range (numpy.ndarray): The smoothed range of x values.
            y_smooth_range (numpy.ndarray): The smoothed range of y values.

        Returns:
            None
        """
        file_path = "data/test_" + self.robotName + "/GMM_learned_actions/" + "GMR_" + str(self.babbled_points) + "_" + self.actionName + '.csv'
        if not os.path.isfile(file_path):
            header = 'time'
            np.savetxt(file_path, x_smooth_range, delimiter=',', header=header, comments='')
            existing_header = [header]
        else:
            existing_header = pd.read_csv(file_path, index_col=0).columns.tolist()
            existing_header.insert(0, 'time')
        existing_data = np.genfromtxt(file_path, delimiter=',',  skip_header=1)  # Skip the header row
        # Check if self.angleId exists in the header
        if self.angleId in existing_header:
            # If self.angleId exists, overwrite the corresponding column
            angle_id_index = np.where(np.array(existing_header) == self.angleId)[0][0]
            existing_data[:, angle_id_index] = y_smooth_range
        else:
            # If self.angleId does not exist, create the header and append the new column
            header_list = list(existing_header)
            header_list.append(self.angleId)
            header = ','.join(header_list)
            existing_data = np.column_stack((existing_data, y_smooth_range))
            np.savetxt(file_path, existing_data, delimiter=',', header=header, comments='')

    def smooth_curve(self, x, y, window_size=10):
        """
        Smooths a curve using a simple moving average.

        Parameters:
        - x (array-like): The x-coordinates of the curve.
        - y (array-like): The y-coordinates of the curve.
        - window_size (int): The size of the moving average window.

        Returns:
        - x_smooth (array-like): The smoothed x-coordinates.
        - y_smooth (array-like): The smoothed y-coordinates.
        """
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        x_smooth = x[:len(y_smooth)]
        return x_smooth, y_smooth

    def plot_results(self, X, Y, avg_signal, means, covariances, index, title):
        """
        Plot the results of the Gaussian Mixture Model.

        Args:
            X (numpy.ndarray): The input data points.
            Y (numpy.ndarray): The cluster labels for each data point.
            avg_signal (numpy.ndarray): The average signal.
            means (list): The means of the Gaussian components.
            covariances (list): The covariances of the Gaussian components.
            index (int): The index of the subplot.
            title (str): The title of the plot.

        Returns:
            None
        """
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            if not np.any(Y == i):
                continue
            plt.scatter(X[Y == i, 0], X[Y == i, 1], color=color)
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        initial = [X[0][0], avg_signal[0]]
        last = [X[-1][0], avg_signal[-1]]

        # Compute cluster centroids
        centroids = np.array([np.nanmean(X[Y == i], axis=0) for i in range(self.n_components)])
        centroids = np.vstack([initial, centroids, last])
        # Remove NaN values if any
        valid_mask = ~np.isnan(centroids[:, 0])
        centroids = centroids[valid_mask]
        if np.isnan(centroids).any():
            centroids = self.impute_nan_values(centroids)

        # Sort centroids based on x-values
        # centroids = np.vstack([X[Y == 0][0], centroids])
        sorted_centroids = centroids[np.argsort(centroids[:, 0])]
        # Plot smooth curve using make_interp_spline for all centroids
        spl = make_interp_spline(sorted_centroids[:, 0], sorted_centroids[:, 1], k=2)
        x_smooth_range = np.linspace(sorted_centroids[:, 0].min(), sorted_centroids[:, 0].max(), 100)
        y_smooth_range = spl(x_smooth_range)
        # self.save_gmr(x_smooth_range, y_smooth_range)
        # Apply moving average to smooth the curve
        plt.plot(x_smooth_range, y_smooth_range, color='red', linestyle='-', linewidth=2, label='Curve without regression - ' + str(self.babbled_points) + " Babbling points")
        plt.xlabel("Normalized Time")
        plt.ylabel("Angle [rad]")
        plt.title("Gaussian Mixture Model " + title)
        plt.grid(True)
        plt.legend()

    def impute_nan_values(self, data):
        """
        Imputes missing values in the given data using the mean strategy.

        Parameters:
            data (array-like): The input data with missing values.

        Returns:
            array-like: The imputed data with missing values replaced by the mean.

        """
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        return imputer.fit_transform(data)

    def test_gmr(self, data, time, title, n_components=3):
        """
        Perform Gaussian Mixture Regression (GMR) on the given data.

        Args:
            data (numpy.ndarray): The data points.
            time (numpy.ndarray): The corresponding time values.
            title (str): The title for the plot.
            n_components (int, optional): The number of Gaussian components. Defaults to 3.

        Returns:
            None
        """
        X = np.ndarray((len(time), 2))
        X[:, 0] = time
        X[:, 1] = data
        n_samples = 1000
        X_test = np.linspace(0, n_samples, n_samples)
        # plt.figure(figsize=(10, 5))
        gmm = GMM(n_components=n_components, random_state=0)
        gmm.from_samples(X)
        Y = gmm.predict(np.array([0]), X_test[:, np.newaxis])
        plt.subplot(2, 1, 2)
        plt.title("Gaussian Mixture Regression " + title)
        plt.scatter(X[:, 0], X[:, 1])
        plot_error_ellipses(plt.gca(), gmm, colors=color_iter)
        plt.plot(X_test, Y.ravel(), c="blue", lw=2, label="Curve with regression - " + str(self.babbled_points) + " Babbling points")
        plt.xlabel("Normalized Time")
        plt.ylabel("Angle [rad]")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig("data/test_" + self.robotName + "/GMM_plots/" + "GMR_" + self.actionName + "_" + self.angleId + "_" + str(self.babbled_points) + '.pdf')
        plt.show()
        self.save_gmr(X_test, Y.ravel())

def arange_time_array(time, arr, chunk_size=100):
    """
    Rearranges the time and array data into chunks of specified size.

    Args:
        time (array-like): The time data.
        arr (array-like): The array data.
        chunk_size (int, optional): The size of each chunk. Defaults to 100.

    Returns:
        numpy.ndarray: A 2D array containing the rearranged time and array data.
    """
    time = np.array(time)
    time = time.reshape(int(time.shape[0]/chunk_size), chunk_size)
    add = np.linspace(0, 0.0001, chunk_size)
    add = np.tile(add, (time.shape[0], 1)).T + time.T
    reordered_time = add.T.reshape(-1).tolist()
    arr = np.array(arr)
    arr = arr.reshape(int(arr.shape[0]/chunk_size), chunk_size)
    reordered_list = arr.reshape(-1).tolist()
    return np.vstack((reordered_time, reordered_list)).T

def eval(time, smoothed_angles, avg_signal, babbled_points, robotName, actionName, angleId, n_components, n_clusters):
    """
    Evaluate and plot Gaussian Mixture Model (GMM) results.

    Args:
        time (array-like): Array of time values.
        smoothed_angles (array-like): Array of smoothed angle values.
        avg_signal (array-like): Array of average signal values.
        babbled_points (array-like): Array of babbled points.
        robotName (str): Name of the robot.
        actionName (str): Name of the action.
        angleId (str): ID of the angle.
        n_components (int): Number of components for the GMM.

    Returns:
        None
    """
    X = arange_time_array(time, smoothed_angles)
    title = " - " + robotName.upper() + " Joint " + angleId + " - " + actionName
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(
        bottom=0.04, top=0.95, hspace=0.2, wspace=0.05, left=0.03, right=0.97
    )
    plotter = GaussianMixturePlotter(robotName, actionName, angleId, n_components, babbled_points)

    # Fit a Gaussian mixture with EM using ten components
    gmm = plotter.fit_gaussian_mixture(X)

    plotter.plot_results(X,
                         gmm.predict(X),
                         avg_signal,
                         gmm.means_,
                         gmm.covariances_,
                         0,
                         title)

    plotter.test_gmr(smoothed_angles, time, title, n_clusters)

if __name__ == "__main__":
    n_samples = 100
    np.random.seed(0)
    X = np.zeros((n_samples, 2))
    step = 4.0 * np.pi / n_samples
    for i in range(X.shape[0]):
        x = i * step
        X[i, 0] = x + np.random.normal(0, 0.1)
        X[i, 1] = 3.0 * (np.sin(x) + np.random.normal(0, 0.2))
    eval(X, n_components=5, n_clusters=5)
