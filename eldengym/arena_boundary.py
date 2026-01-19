"""Arena boundary detection and distance calculations using SDF."""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt


@dataclass
class BoundaryDistances:
    """Distances from a point to arena boundaries."""
    inside: bool
    nearest: float  # Distance to nearest boundary (negative if outside)
    north: float    # Distance to boundary going up (+Y direction)
    south: float    # Distance to boundary going down (-Y direction)
    east: float     # Distance to boundary going right (+X direction)
    west: float     # Distance to boundary going left (-X direction)


class ArenaBoundary:
    """
    Arena boundary representation using Signed Distance Field (SDF).

    Provides fast O(1) queries for:
    - Point inside/outside test
    - Distance to nearest boundary
    - Distance to boundary in 4 cardinal directions
    """

    def __init__(
        self,
        polygon: Polygon,
        resolution: float = 0.5,
        padding: float = 10.0,
    ):
        """
        Initialize arena boundary from a polygon.

        Args:
            polygon: Shapely Polygon representing the arena boundary
            resolution: Grid cell size (smaller = more accurate, more memory)
            padding: Extra space around polygon for SDF computation
        """
        self.polygon = polygon
        self.resolution = resolution
        self.padding = padding

        # Compute bounding box
        minx, miny, maxx, maxy = polygon.bounds
        self.x_min = minx - padding
        self.x_max = maxx + padding
        self.y_min = miny - padding
        self.y_max = maxy + padding

        # Create grid
        self.nx = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.ny = int(np.ceil((self.y_max - self.y_min) / resolution))

        # Compute SDF
        self._compute_sdf()

        # Create interpolator for fast queries
        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        self._sdf_interp = RegularGridInterpolator(
            (x, y), self.sdf, method='linear', bounds_error=False, fill_value=np.inf
        )

    def _compute_sdf(self):
        """Compute the signed distance field."""
        # Create coordinate grids
        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # Compute inside/outside mask
        inside_mask = np.zeros((self.nx, self.ny), dtype=bool)
        boundary = self.polygon.boundary

        for i in range(self.nx):
            for j in range(self.ny):
                point = Point(xx[i, j], yy[i, j])
                inside_mask[i, j] = self.polygon.contains(point)

        # Compute distance transform for inside and outside
        # Distance transform gives distance to nearest False cell
        dist_inside = ndimage.distance_transform_edt(inside_mask) * self.resolution
        dist_outside = ndimage.distance_transform_edt(~inside_mask) * self.resolution

        # SDF: negative inside, positive outside
        self.sdf = np.where(inside_mask, -dist_inside, dist_outside)
        self.inside_mask = inside_mask

    def query(self, x: float, y: float) -> BoundaryDistances:
        """
        Query boundary distances for a point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            BoundaryDistances with inside/outside status and distances
        """
        # Get SDF value (negative = inside)
        sdf_value = float(self._sdf_interp([[x, y]])[0])
        inside = sdf_value < 0
        nearest = abs(sdf_value)

        # Ray-cast in 4 directions to find boundary distances
        point = Point(x, y)

        north = self._raycast_distance(x, y, 0, 1)   # +Y
        south = self._raycast_distance(x, y, 0, -1)  # -Y
        east = self._raycast_distance(x, y, 1, 0)    # +X
        west = self._raycast_distance(x, y, -1, 0)   # -X

        return BoundaryDistances(
            inside=inside,
            nearest=nearest if inside else -nearest,
            north=north,
            south=south,
            east=east,
            west=west,
        )

    def query_batch(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query multiple points at once (faster than individual queries).

        Args:
            points: Array of shape (N, 2) with x, y coordinates

        Returns:
            Tuple of (inside_mask, nearest_distances)
        """
        sdf_values = self._sdf_interp(points)
        inside = sdf_values < 0
        nearest = np.abs(sdf_values)
        return inside, nearest

    def _raycast_distance(self, x: float, y: float, dx: int, dy: int, max_dist: float = 1000.0) -> float:
        """
        Cast a ray and find distance to boundary.

        Args:
            x, y: Starting point
            dx, dy: Direction (should be -1, 0, or 1)
            max_dist: Maximum distance to search

        Returns:
            Distance to boundary in that direction (inf if not found)
        """
        # Create ray as a line
        end_x = x + dx * max_dist
        end_y = y + dy * max_dist
        ray = LineString([(x, y), (end_x, end_y)])

        # Find intersection with polygon boundary
        intersection = ray.intersection(self.polygon.boundary)

        if intersection.is_empty:
            return float('inf')

        # Get nearest intersection point
        if intersection.geom_type == 'Point':
            return Point(x, y).distance(intersection)
        elif intersection.geom_type == 'MultiPoint':
            return min(Point(x, y).distance(p) for p in intersection.geoms)
        elif intersection.geom_type == 'LineString':
            # Ray grazes the boundary
            return Point(x, y).distance(Point(intersection.coords[0]))
        else:
            # GeometryCollection or other
            return Point(x, y).distance(intersection)

    def is_inside(self, x: float, y: float) -> bool:
        """Quick inside/outside test."""
        return float(self._sdf_interp([[x, y]])[0]) < 0

    def nearest_distance(self, x: float, y: float) -> float:
        """Get signed distance to boundary (negative = inside)."""
        return float(self._sdf_interp([[x, y]])[0])

    def get_sdf_normal(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get the SDF gradient normal at a point.

        The normal points in the direction of increasing SDF (towards outside).
        For a point inside the arena, this points towards the nearest boundary.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Tuple of (normal_x, normal_y), normalized to unit length
        """
        # Compute gradient using finite differences
        eps = self.resolution
        sdf_px = float(self._sdf_interp([[x + eps, y]])[0])
        sdf_mx = float(self._sdf_interp([[x - eps, y]])[0])
        sdf_py = float(self._sdf_interp([[x, y + eps]])[0])
        sdf_my = float(self._sdf_interp([[x, y - eps]])[0])

        grad_x = (sdf_px - sdf_mx) / (2 * eps)
        grad_y = (sdf_py - sdf_my) / (2 * eps)

        # Normalize to unit vector
        mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        if mag > 1e-6:
            return (grad_x / mag, grad_y / mag)
        else:
            return (0.0, 0.0)

    def query_sdf(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Query SDF value and normal in one call.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Tuple of (sdf_value, normal_x, normal_y)
            - sdf_value: negative inside, positive outside
            - normal_x, normal_y: unit vector pointing towards boundary
        """
        sdf_value = self.nearest_distance(x, y)
        normal_x, normal_y = self.get_sdf_normal(x, y)
        return (sdf_value, normal_x, normal_y)

    def save(self, path: str):
        """Save boundary to file."""
        data = {
            'polygon': list(self.polygon.exterior.coords),
            'resolution': self.resolution,
            'padding': self.padding,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ArenaBoundary':
        """Load boundary from file."""
        with open(path) as f:
            data = json.load(f)
        polygon = Polygon(data['polygon'])
        return cls(polygon, data['resolution'], data['padding'])

    @classmethod
    def from_path_data(
        cls,
        path_data_file: str,
        simplify_tolerance: float = 1.0,
        resolution: float = 0.5,
        use_convex_hull: bool = False,
    ) -> 'ArenaBoundary':
        """
        Create arena boundary from traced path data.

        Args:
            path_data_file: Path to JSON file from trace_paths.py
            simplify_tolerance: Tolerance for polygon simplification
            resolution: SDF grid resolution
            use_convex_hull: Use convex hull instead of traced polygon

        Returns:
            ArenaBoundary instance
        """
        # Load path data
        with open(path_data_file) as f:
            data = json.load(f)

        # Extract player path coordinates (using same transform as visualization)
        player_path = data.get('player_path', [])
        if not player_path:
            raise ValueError("No player path data found")

        # Transform coordinates (same as trace_paths.py visualization)
        coords = [(p['global_y'], -p['global_x']) for p in player_path]

        if use_convex_hull:
            from shapely.geometry import MultiPoint
            points = MultiPoint(coords)
            polygon = points.convex_hull
        else:
            # Create polygon from path (assuming it traces the boundary)
            # Close the polygon if not closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polygon = Polygon(coords)

            # Simplify to reduce noise
            polygon = polygon.simplify(simplify_tolerance)

            # Ensure valid polygon
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

        return cls(polygon, resolution=resolution)

    def visualize(
        self,
        output_path: Optional[str] = None,
        show_sdf: bool = True,
        test_points: Optional[List[Tuple[float, float]]] = None,
    ) -> plt.Figure:
        """
        Visualize the arena boundary and SDF.

        Args:
            output_path: Where to save figure
            show_sdf: Show SDF heatmap
            test_points: Optional list of points to show distances for
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        if show_sdf:
            # Plot SDF as heatmap
            x = np.linspace(self.x_min, self.x_max, self.nx)
            y = np.linspace(self.y_min, self.y_max, self.ny)

            # Transpose for correct orientation
            im = ax.imshow(
                self.sdf.T, origin='lower',
                extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                cmap='RdBu', alpha=0.7,
                vmin=-np.percentile(np.abs(self.sdf), 95),
                vmax=np.percentile(np.abs(self.sdf), 95),
            )
            plt.colorbar(im, ax=ax, label='Signed Distance (negative=inside)')

        # Plot polygon boundary
        x_poly, y_poly = self.polygon.exterior.xy
        ax.plot(x_poly, y_poly, 'k-', linewidth=2, label='Boundary')

        # Plot test points if provided
        if test_points:
            for px, py in test_points:
                result = self.query(px, py)
                color = 'green' if result.inside else 'red'
                ax.scatter([px], [py], c=color, s=100, zorder=5)

                # Draw rays to boundaries
                ax.annotate(
                    f'N:{result.north:.1f}\nS:{result.south:.1f}\nE:{result.east:.1f}\nW:{result.west:.1f}',
                    (px, py), fontsize=8, ha='left'
                )

        # Labels match trace_paths convention (Y horizontal, X vertical)
        ax.set_xlabel('Global Y (horizontal)')
        ax.set_ylabel('Global X (vertical)')
        ax.set_title('Arena Boundary SDF')
        ax.set_aspect('equal')
        ax.legend()

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig


def main():
    """CLI for arena boundary operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Arena boundary tools")
    subparsers = parser.add_subparsers(dest="command")

    # Create from path data
    create_parser = subparsers.add_parser("create", help="Create boundary from path data")
    create_parser.add_argument("input", help="Path data JSON file")
    create_parser.add_argument("--output", "-o", default="arena_boundary.json", help="Output file")
    create_parser.add_argument("--resolution", type=float, default=0.5, help="SDF resolution")
    create_parser.add_argument("--simplify", type=float, default=1.0, help="Simplification tolerance")
    create_parser.add_argument("--convex", action="store_true", help="Use convex hull")
    create_parser.add_argument("--visualize", action="store_true", help="Show visualization")

    # Query a point
    query_parser = subparsers.add_parser("query", help="Query a point")
    query_parser.add_argument("boundary", help="Boundary JSON file")
    query_parser.add_argument("x", type=float)
    query_parser.add_argument("y", type=float)

    # Visualize
    viz_parser = subparsers.add_parser("visualize", help="Visualize boundary")
    viz_parser.add_argument("boundary", help="Boundary JSON file")
    viz_parser.add_argument("--output", "-o", help="Output image")

    args = parser.parse_args()

    if args.command == "create":
        print(f"Creating boundary from {args.input}...")
        boundary = ArenaBoundary.from_path_data(
            args.input,
            simplify_tolerance=args.simplify,
            resolution=args.resolution,
            use_convex_hull=args.convex,
        )
        boundary.save(args.output)
        print(f"Saved to {args.output}")

        if args.visualize:
            boundary.visualize()
            plt.show()

    elif args.command == "query":
        boundary = ArenaBoundary.load(args.boundary)
        result = boundary.query(args.x, args.y)
        print(f"Point ({args.x}, {args.y}):")
        print(f"  Inside: {result.inside}")
        print(f"  Nearest: {result.nearest:.2f}")
        print(f"  North: {result.north:.2f}")
        print(f"  South: {result.south:.2f}")
        print(f"  East: {result.east:.2f}")
        print(f"  West: {result.west:.2f}")

    elif args.command == "visualize":
        boundary = ArenaBoundary.load(args.boundary)
        boundary.visualize(args.output)
        if not args.output:
            plt.show()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
