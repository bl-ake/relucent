import multiprocessing as mp
import os
import random
from collections import defaultdict
from functools import partial

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .bvs import BVManager, BVPriorityQueue
from .poly import Polyhedron
from .utils import BlockingQueue, NonBlockingQueue, get_colors, get_env


def set_globals(get_net, get_volumes=True):
    global env
    env = get_env()
    global net
    net = get_net
    global dim
    dim = np.prod(net.input_shape)
    global get_vol_calc
    get_vol_calc = get_volumes


def poly_calculations(task, **kwargs):
    bv = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    p = Polyhedron(net, bv)

    try:
        p._halfspaces, p._W, p._b = p.get_hs()
        p.get_center_inradius(env=env)
        p.get_interior_point(env=env)
        p._interior_point_norm = np.linalg.norm(p.interior_point).item()
        p._Wl2 = np.linalg.norm(p.W).item()

        if dim <= 6 and get_vol_calc:
            p.volume
        if p._shis is None:
            if "collect_info" in kwargs:
                p._shis, shi_info = p.get_shis(env=env, **kwargs)
            else:
                p._shis = p.get_shis(env=env, **kwargs)
    except ValueError as error:
        return error, *rest
    p.clean_data()
    random.shuffle(p._shis)
    return (p, *rest)


def get_ip(p, shi):
    try:
        bv = p.bv.copy()
        bv[0, shi] = -bv[0, shi]
        n = Polyhedron(net, bv)
        for max_radius in [0.01, 0.1, 1, 10, 100]:
            try:
                n.get_interior_point(env=env, max_radius=max_radius)
            except ValueError:
                print("Increasing max radius to find interior point")
        return n, shi
    except ValueError as e:
        return e, None


def astar_calculations(task, **kwargs):
    p = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    p.net = net

    if p._inradius is None:
        p.get_center_inradius(env=env)
    if p._interior_point is None:
        p.get_interior_point(env=env)

    try:
        if p._shis is None:
            if "collect_info" in kwargs:
                p._shis, shi_info = p.get_shis(env=env, **kwargs)
            else:
                p._shis = p.get_shis(env=env, **kwargs)
    except Exception as error:
        return p, error, *rest
    p.clean_data()
    random.shuffle(p._shis)
    # p, neighbors = get_neighbors(p, (shi for shi in p._shis if shi != task[1]))
    return p, *rest


class Complex:
    def __init__(self, net):
        self.net = net

        ## TODO: Try replacing with just a dictionary that incremements by 1
        self.bvm = BVManager()
        self.index2poly = list()

        net_layers = list(net.layers.values())
        self.bv_layers = [
            i
            for i, (layer, next_layer) in enumerate(zip(net_layers[:-1], net_layers[1:]))
            if isinstance(next_layer, nn.ReLU)
        ]

        x = torch.zeros((1,) + net.input_shape, device=net.device, dtype=net.dtype)
        self.bvi2maski = []
        for i, layer in enumerate(self.net.layers.values()):
            x = layer(x)
            if i in self.bv_layers:
                it = np.nditer(x.detach().cpu().numpy(), flags=["multi_index"])
                for _ in it:
                    self.bvi2maski.append((i, it.multi_index))

    def __getitem__(self, key):
        if isinstance(key, Polyhedron):
            return self.index2poly[self.bvm[key.bv]]
        elif isinstance(key, int):
            raise NotImplementedError
        elif isinstance(key, (np.ndarray, torch.Tensor)):
            return self.index2poly[self.bvm[key]]
        else:
            raise KeyError(f"Polyhedron with key {key} not in Complex")

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __iter__(self):
        for p in self.index2poly:
            yield p

    def __len__(self):
        return len(self.index2poly)

    # def __reduce__(self):
    # raise ValueError("Don't Pickle Complexes!")

    @property
    def dim(self):
        return np.prod(self.net.input_shape)

    @torch.no_grad()
    def bv_iterator(self, batch):
        x = (
            torch.tensor(batch, device=self.net.device, dtype=self.net.dtype)
            if isinstance(batch, np.ndarray)
            else batch
        ).reshape((-1, *self.net.input_shape))
        for i, layer in enumerate(self.net.layers.values()):
            x = layer(x)
            if i in self.bv_layers:
                yield torch.sign(x)  # * (torch.abs(x) < 1e-12)
                if i == self.bv_layers[-1]:
                    break

    def point2bv(self, batch, numpy=False):
        result = torch.hstack(list(self.bv_iterator(batch)))
        return result.detach().cpu().numpy() if numpy else result

    def point2poly(self, point, check_exists=True, numpy=False):
        bv = self.point2bv(point, numpy=numpy)
        return self.bv2poly(bv, check_exists=check_exists)

    def bv2poly(self, bv, check_exists=True):
        if check_exists and bv in self:
            return self[bv]
        else:
            return Polyhedron(self.net, bv)

    def add_bv(self, bv, check_exists=True):
        p = self.bv2poly(bv, check_exists=check_exists)
        p = self.add_polyhedron(p)
        return p

    def add_polyhedron(self, p, overwrite=False):
        if p not in self:
            self.index2poly.append(p)
            self.bvm.add(p.bv)
        elif p in self and overwrite:
            self.index2poly[self.bvm[p.bv]] = p
        return self[p]

    def add_point(self, data, numpy=False):
        p = self.point2poly(data, numpy=numpy)
        p = self.add_polyhedron(p)
        return p

    def clean_data(self):
        for poly in self:
            poly.clean_data()

    @torch.no_grad()
    def adjacent_polyhedra(self, poly):
        ps = set()
        shis = poly.shis
        for shi in shis:
            if poly.bv[0, shi] == 0:
                continue
            bv = poly.bv.clone()
            bv[0, shi] = -bv[0, shi]
            self.add_bv(bv)
        return ps

    def parallel_add(self, points, nworkers=None, bound=1e5, **kwargs):
        nworkers = nworkers or os.process_cpu_count()
        print(f"Running on {nworkers} workers")

        with mp.Pool(nworkers, initializer=set_globals, initargs=(self.net,)) as pool:
            bvs = list(map(partial(self.point2bv, numpy=True), tqdm(points, desc="Getting BVs", mininterval=5)))
            ps = pool.map(
                partial(poly_calculations, bound=bound, **kwargs), tqdm(bvs, desc="Adding Polys", mininterval=5)
            )
            ps = [p[0] if isinstance(p[0], Polyhedron) else None for p in ps]
            for p in ps:
                if p is not None:
                    self.add_polyhedron(p)
            return ps

    def searcher(
        self,
        start=None,
        max_depth=float("inf"),
        max_polys=float("inf"),
        queue=None,
        bound=1e5,
        nworkers=None,
        get_volumes=True,
        verbose=1,
        **kwargs,
    ):
        found_bvs = BVManager()
        nworkers = nworkers or os.process_cpu_count()
        ## NOTE: If nworkers>1, the traversal order may not be correct
        # if len(self) > 0:
        #     raise ValueError("Complex already has polyhedra")
        if verbose:
            print(f"Running on {nworkers} workers")
        if queue is None:
            queue = BlockingQueue()
        if start is None:
            start = self.add_point(torch.zeros(self.net.input_shape, device=self.net.device, dtype=self.net.dtype))
        elif isinstance(start, torch.Tensor):
            start = self.add_point(start)
        start.bv = start.bv.detach().cpu().numpy()
        self.add_bv(start.bv)
        found_bvs.add(start.bv)
        if (start.bv == 0).any():
            raise ValueError("Start point must not be on a hyperplane")
        start._shis = start.get_shis(bound=bound, **kwargs)
        ##TODO:
        ## replace with something like queue.push((start.bv, None, 0, self.bvm[start.bv]))
        for shi in start.shis:
            new_bv = start.bv.copy()
            new_bv[0, shi] *= -1
            found_bvs.add(new_bv)
            queue.push((new_bv, shi, 1, self.bvm[start.bv]))
            assert new_bv in found_bvs

        rolling_average = len(start.shis)
        bad_shi_computations = []
        pbar = tqdm(
            desc="Search Progress",
            mininterval=5,
            total=max_polys if max_polys != float("inf") else None,
            disable=not verbose,
        )
        pbar.update(n=1)
        pbar.get_lock().locks = []

        unprocessed = len(queue)
        depth = 0

        with mp.Pool(nworkers, initializer=set_globals, initargs=(self.net, get_volumes)) as pool:
            try:
                for p, shi, depth, node_index in pool.imap_unordered(
                    partial(poly_calculations, bound=bound, **kwargs), queue
                ):
                    unprocessed -= 1
                    node = self.index2poly[node_index]
                    if not isinstance(p, Polyhedron):
                        bad_shi_computations.append((node, shi, depth, str(p)))
                        node._shis.remove(shi)
                        continue

                    p.net = self.net

                    p = self.add_polyhedron(p)

                    if depth < max_depth:
                        for new_shi in p.shis:
                            if new_shi != shi and len(self) < max_polys:
                                bv = p.bv.copy()
                                bv[0, new_shi] *= -1
                                if bv not in found_bvs:
                                    queue.push((bv, new_shi, depth + 1, self.bvm[p.bv]))
                                    found_bvs.add(bv)
                                    unprocessed += 1

                    pbar.update(n=len(self) - pbar.n)
                    rolling_average = (rolling_average * (len(self) - 1) + len(p.shis)) / len(self)
                    pbar.set_postfix_str(
                        f"Depth: {depth}  Unprocessed: {unprocessed}  Faces: {len(p._shis)}  Avg: {rolling_average:.2f} IP Norm: {p._interior_point_norm or -1:.2f}  Finite: {p._finite} Mistakes: {len(bad_shi_computations)}",
                        refresh=False,
                    )

                    if unprocessed == 0 or len(self) >= max_polys:
                        break
            except Exception:
                raise
            finally:
                queue.close()
                pbar.close()

        search_info = {
            "Search Depth": depth,
            "Avg # Facets Uncorrected": rolling_average,
            "Search Time": pbar.format_dict["elapsed"],
            "Bad SHI Computations": bad_shi_computations,
            "Complete": unprocessed == 0,
        }

        return search_info

    def bfs(self, **kwargs):
        return self.searcher(**kwargs)

    def dfs(self, **kwargs):
        return self.searcher(queue=BlockingQueue(pop=lambda x: x.popleft()), **kwargs)

    def random_walk(self, **kwargs):
        return self.searcher(
            queue=list(),
            pop=lambda x: x.pop(random.randrange(0, len(x) - 1)),
            **kwargs,
        )

    def greedy_path_helper(self, start, end, diffs=None):
        if start == end:
            # print("Start and end points are the same")
            return [start]

        if (start.bv == 0).any():
            raise ValueError("Start point must not be on a hyperplane")

        diffs = diffs or set(np.argwhere((start.bv != end.bv).flatten()).flatten().tolist())

        print("Diffs:", diffs)

        if not start._shis:
            try:
                start.get_shis()
            except ValueError:
                return None
        groupa = set(start.shis) & diffs
        groupb = set(start.shis) - diffs
        for shi in list(groupa):
            print("Crossing", shi)
            new_bv = start.bv.copy()
            new_bv[0, shi] *= -1
            next_poly = self.bv2poly(new_bv)
            rest = self.greedy_path_helper(next_poly, end, diffs - {shi})
            if rest is not None:
                return [start] + rest
        for shi in list(groupb):
            print("Crossing", shi)
            new_bv = start.bv.copy()
            new_bv[0, shi] *= -1
            next_poly = self.bv2poly(new_bv)
            rest = self.greedy_path_helper(next_poly, end, diffs + {shi})
            if rest is not None:
                return [start] + rest
        return None

    def greedy_path(self, start, end):
        start = self.add_point(start, numpy=True)
        end = self.add_point(end, numpy=True)
        return self.greedy_path_helper(start, end)

    def hamming_astar(self, start, end, nworkers=1, bound=1e5, max_polys=float("inf"), show_pbar=True, **kwargs):
        # from hanging_threads import start_monitoring

        # start_monitoring(seconds_frozen=10, test_interval=100)

        start = self.add_point(start, numpy=True)
        end = self.add_point(end, numpy=True)
        if start == end:
            print("Start and end points are in the same region")
            start.get_center_inradius()
            start.get_interior_point()
            start.get_shis(bound=bound)
            return [start]

        if (start.bv == 0).any():
            raise ValueError("Start point must not be on a hyperplane")

        cameFrom = dict()
        gScore = defaultdict(lambda: float("inf"))
        fScore = defaultdict(lambda: float("inf"))

        openSet = NonBlockingQueue(
            queue_class=BVPriorityQueue,
            push=lambda pq, task, priority: pq.push(task, priority=priority),
            pop=lambda pq: pq.pop(),
        )
        # found_bvs = BVManager()
        # nworkers = nworkers or os.process_cpu_count()

        gScore[start] = 0
        fScore[start] = (start.bv != end.bv).sum()
        # found_bvs.add(start.bv)
        # found = set(start)

        start._shis = start.get_shis(bound=bound, **kwargs)

        openSet.push((start, None, 0), 0)

        bad_shi_computations = []
        pbar = tqdm(
            desc="Search Progress" + (str(show_pbar) if show_pbar is not True else ""),
            mininterval=1,
            leave=True,
            total=max_polys if max_polys != float("inf") else None,
            disable=not show_pbar,
        )
        pbar.update(n=1)

        unprocessed = len(openSet)
        depth = 0
        neighbor = None
        min_dist = float("inf")
        min_p = None

        def heuristic(p, depth, shi):
            hamming = (p.bv != end.bv).sum()
            dist = np.linalg.norm(p.interior_point - end.interior_point).item()
            # bias = -1 / ((1 + dist) ** 0.1)  ## TODO: Test if this is faster
            # bias = -1 / (1 + np.log(dist + 10))
            bias = -1 / (1 + dist)
            # bias = 0
            # bias = 1 / (1 + depth) - 1
            return hamming + 0.9 * bias

        def d(p1, p2):
            return 1

        # with mp.Pool(nworkers, initializer=set_globals, initargs=(self.net,)) as pool:
        pool = mp.Pool(nworkers, initializer=set_globals, initargs=(self.net,))
        try:
            # for p, neighbors, shi, depth, node_index in pool.imap(
            #     partial(astar_calculations, bound=bound, **kwargs), openSet
            # ):
            set_globals(self.net)
            for item in map(partial(astar_calculations, bound=bound, **kwargs), openSet):
                # found.remove(item[0])
                if isinstance(item[1], Exception):
                    bad_shi_computations.append(item)
                    continue
                unprocessed -= 1

                p, shi, depth = item

                for neighbor, neighbor_shi in pool.imap_unordered(
                    partial(get_ip, p), (i for i in p.shis if isinstance != shi), chunksize=32
                ):
                    # for neighbor, neighbor_shi in map(
                    #     partial(get_ip, p),
                    #     (i for i in p.shis if i != shi),
                    # ):
                    if not isinstance(neighbor, Polyhedron):
                        p._shis.remove(neighbor)

                    tentative_gScore = gScore[p] + d(p, neighbor)
                    neighbor.net = self.net
                    if tentative_gScore < gScore[neighbor]:  ## Only needed with an inconsistent heuristic
                        cameFrom[neighbor] = p
                        gScore[neighbor] = tentative_gScore
                        dist = heuristic(neighbor, depth, shi)
                        fScore[neighbor] = tentative_gScore + dist
                        # options.append(
                        #     {
                        #         "neighbor": neighbor,
                        #         "neighbor_shi": neighbor_shi,
                        #         "fScore": fScore[neighbor],
                        #         "improvement": neighbor.bv[0, neighbor_shi] == end.bv[0, neighbor_shi],
                        #     }
                        # )
                        if dist < min_dist:
                            min_dist = dist
                            min_p = neighbor
                        openSet.push((neighbor, neighbor_shi, depth + 1), fScore[neighbor])
                        unprocessed += 1
                        if neighbor == end:
                            break
                    if neighbor == end:
                        break

                # next_one = openSet.deque.pq[0][2]
                # new_hamming = (next_one.bv != end.bv).sum()
                # if new_hamming >= (p.bv != end.bv).sum():
                #     p2 = next_one
                #     p1 = pd.DataFrame(options).sort_values("fScore", ascending=True).iloc[0]["neighbor"]

                #     print(
                #         f"The next polyhedron to be searched is {p2}: fScore: {fScore[p2]} gScore: {gScore[p2]} Hamming: {(p2.bv != end.bv).sum()} Heuristic: {heuristic(p2, 0, 0)} | Distance: {np.linalg.norm(p2.interior_point - end.interior_point).item()}"
                #     )
                #     print(
                #         f"A better option could be: {p1}: fScore: {fScore[p1]} gScore: {gScore[p1]} Hamming: {(p1.bv != end.bv).sum()} Heuristic: {heuristic(p1, 0, 0)} | Distance: {np.linalg.norm(p1.interior_point - end.interior_point).item()}"
                #     )

                pbar.update(n=len(cameFrom) - pbar.n)
                pbar.set_postfix_str(
                    f"Min Distance: {min_dist:.3f} Depth: {depth} Open Set: {unprocessed} Mistakes: {len(bad_shi_computations)} | Finite: {p.finite} # SHIs: {len(p.shis)}",
                    refresh=False,
                )

                if min_dist < 1:
                    if 0 < min_dist:
                        last_shi = np.argwhere((min_p.bv != end.bv).flatten()).item()
                        if last_shi in min_p.shis:
                            cameFrom[end] = min_p
                            neighbor = end
                            break
                    else:
                        # raise ValueError("what in tarnation???")
                        neighbor = end
                        break

                if unprocessed == 0 or len(cameFrom) >= max_polys:
                    break
        except Exception:
            raise
        finally:
            # print(f"Closing out after {pbar.n} iterations")
            pool.close()
            pool.terminate()

            openSet.close()
            tqdm.get_lock().locks = []
            pbar.close()
        #     print("Closed out")
        # print("Finished A* Search")
        if neighbor == end:
            path = [end]
            while path[-1] != start:
                assert cameFrom[path[-1]] not in path, path
                path.append(cameFrom[path[-1]])
            path.reverse()
            [(p.Wl2, p.inradius) for p in path]
            # print(f"Path found with length {len(path) - 1}:")
            # if len(path) < 100:
            #     for p1, p2 in zip(path[:-1], path[1:]):
            #         print(f"    {p1} - {np.argwhere((p1.bv != p2.bv).flatten()).item()} -> {p2}")
            return path
        else:
            # print(f"No Path Found - Final Distance: {min_dist - 1}")
            return None

    def get_poly_attrs(self, attrs):
        return {attr: [getattr(poly, "_".join(attr.lower().split("_"))) for poly in self] for attr in attrs}

    def get_dual_graph(
        self, relabel=False, plot=False, node_color=None, node_size=None, cmap="viridis", match_locations=False
    ):
        G = nx.Graph()
        for poly in self:
            G.add_node(poly, label=str(poly))
        for poly in tqdm(self, desc="Creating Dual Graph", leave=False):
            bv = poly.bv
            for shi in poly.shis:
                bv[0, shi] *= -1
                if bv in self:
                    G.add_edge(poly, self[bv], shi=shi)
                bv[0, shi] *= -1
        if plot:
            if match_locations:
                if self.dim != 2:
                    raise ValueError("Polyhedra must be 2D to match locations")

                nx.set_node_attributes(G, False, "physics")
                assert poly.interior_point is not None
                nx.set_node_attributes(
                    G,
                    {poly: poly.interior_point[0].item() * 10 for poly in G.nodes},
                    "x",
                )
                nx.set_node_attributes(
                    G,
                    {poly: poly.interior_point[1].item() * 10 for poly in G.nodes},
                    "y",
                )

            if node_color == "Wl2":
                colors = get_colors([poly.Wl2 for poly in G.nodes], cmap=cmap)
                for c, poly in zip(colors, G.nodes):
                    G.nodes[poly]["color"] = c
            elif node_color == "volume":
                colors = get_colors([poly.ch.volume for poly in G.nodes], cmap=cmap)
                for c, poly in zip(colors, G.nodes):
                    G.nodes[poly]["color"] = c

            if node_size == "volume":
                sizes = [poly.ch.volume for poly in G.nodes]
                maxsize = max(sizes)
                for size, poly in zip(sizes, G.nodes):
                    G.nodes[poly]["size"] = (10 + 1000 * size / maxsize) ** 1
            else:
                nx.set_node_attributes(G, 4, "size")

            for node in G.nodes:
                G.nodes[node]["label"] = " "
                G.nodes[node]["title"] = str(node)
            for edge in G.edges:
                G.edges[edge]["label"] = " "
                G.edges[edge]["title"] = str(G.edges[edge]["shi"])
        if plot or relabel:
            G = nx.relabel_nodes(G, {poly: i for i, poly in enumerate(self)})
        return G

    def recover_from_dual_graph(self, G, initial_bv, source=0):
        G = G.copy()
        initial_p = self.add_bv(initial_bv)
        G.nodes[source]["poly"] = initial_p
        for edge in tqdm(nx.edge_bfs(G, source=0), desc="Recovering Polyhedra", total=G.number_of_edges()):
            poly1, shi = G.nodes[edge[0]]["poly"], G.edges[edge]["shi"]
            poly2_bv = poly1.bv.clone()
            assert poly2_bv[0, shi] != 0
            poly2_bv[0, shi] *= -1
            poly2 = self.add_bv(poly2_bv)

            G.nodes[edge[1]]["poly"] = poly2

        for node in G:
            self[G.nodes[node]["poly"]]._shis = [G.edges[edge]["shi"] for edge in G.edges(node)]

        return G

    def plot(self, label_regions=False, color=None, highlight_regions=None, bv_name=False, **kwargs):
        fig = go.Figure()
        polys = list(self)
        if color == "Wl2":
            colors = get_colors([poly.Wl2 for poly in polys])
        else:
            color_scheme = px.colors.qualitative.Plotly
            try:
                coloring = nx.algorithms.coloring.equitable_color(self.get_dual_graph(), len(color_scheme))
                remap, idx = dict(), 0
                for p in polys:
                    if coloring[p] not in remap:
                        remap[coloring[p]] = idx
                        idx += 1
                colors = [color_scheme[remap[coloring[i]]] for i in polys]
            except Exception:
                print("Could not find equitable coloring, using random colors")
                colors = [color_scheme[i % len(color_scheme)] for i in range(len(polys))]
        for c, poly in tqdm(zip(colors, polys), desc="Plotting Polyhedra", total=len(polys)):
            if (highlight_regions is not None) and ((poly in highlight_regions) or (str(poly) in highlight_regions)):
                c = "red"
            p_plot = poly.plot2d(
                name=f"{poly.bv.flatten().astype(int).tolist()}" if bv_name else f"{poly}",
                fillcolor=c,
                line_color="black",
                mode="lines",  ## Comment out to mouse over intersections
                **kwargs,
            )
            if p_plot is not None:
                fig.add_trace(p_plot)
            if label_regions and poly.center is not None:
                fig.add_trace(
                    go.Scatter(x=[poly.center[0]], y=[poly.center[1]], mode="text", text=str(poly), showlegend=False)
                )
        interior_points = [np.max(np.abs(p.interior_point)) for p in self if p.finite]
        maxcoord = (np.max(interior_points) * 1.1) if len(interior_points) > 0 else 10
        # maxcoord = 10
        fig.update_layout(
            showlegend=True,
            # xaxis = dict(visible=False),
            # yaxis = dict(visible=False),
            plot_bgcolor="white",
            xaxis=dict(range=(-maxcoord, maxcoord)),
            yaxis=dict(range=(-maxcoord, maxcoord)),
        )
        return fig

    def plot3d(
        self,
        label_regions=False,
        color=None,
        highlight_regions=None,
        show_axes=False,
        project=True,
        **kwargs,
    ):
        fig = go.Figure()
        polys = list(self)
        if color == "Wl2":
            colors = get_colors([poly.Wl2 for poly in polys])
        else:
            color_scheme = px.colors.qualitative.Plotly
            try:
                coloring = nx.algorithms.coloring.equitable_color(self.get_dual_graph(), len(color_scheme))
                colors = [color_scheme[coloring[i]] for i in polys]
            except Exception:
                print("Could not find equitable coloring, using random colors")
                colors = [color_scheme[i % len(color_scheme)] for i in range(len(polys))]
        outlines, meshes = [], []
        for c, poly in tqdm(zip(colors, polys), desc="Plotting Polyhedra", total=len(polys)):
            if (highlight_regions is not None) and ((poly in highlight_regions) or (str(poly) in highlight_regions)):
                c = "red"
            p_plot = poly.plot3d(
                name=f"{poly}",
                color=c,
                # outlinecolor="black",
                **kwargs,
            )
            if p_plot is not None:
                if isinstance(p_plot, dict):
                    if "mesh" in p_plot:
                        meshes.append(p_plot["mesh"])
                    if "outline" in p_plot:
                        outlines.append(p_plot["outline"])
                else:
                    fig.add_trace(p_plot)
            if project is not None:
                p_plot = poly.plot3d(
                    name=f"{poly}",
                    color=c,
                    project=project,
                    **kwargs,
                )
                if p_plot is not None:
                    if isinstance(p_plot, dict):
                        if "mesh" in p_plot:
                            meshes.append(p_plot["mesh"])
                        if "outline" in p_plot:
                            outlines.append(p_plot["outline"])
                    else:
                        fig.add_trace(p_plot)
            if label_regions and poly.center is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[poly.center[0]],
                        y=[poly.center[1]],
                        z=[
                            self.net(torch.tensor(poly.center, device=self.net.device, dtype=self.net.dtype).T)
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                            .flatten()[:, 0]
                        ],
                        mode="text",
                        text=str(poly),
                        showlegend=False,
                    )
                )
        for outline in outlines:
            fig.add_trace(outline)
        for mesh in meshes:
            fig.add_trace(mesh)
        maxcoord = np.median([np.max(np.abs(p.interior_point)) for p in self if p.finite]) * 1.1
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=(-maxcoord, maxcoord), visible=show_axes),
                yaxis=dict(range=(-maxcoord, maxcoord), visible=show_axes),
                zaxis=dict(visible=show_axes),
            ),
        )
        return fig
