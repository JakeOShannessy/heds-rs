use slab::Slab;
mod point;
pub use point::*;

/// Offset into the vertex array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HVertexRef(usize);

/// Offset into the edge array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HEdgeRef(usize);

/// Offset into the face array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HFaceRef(usize);

/// This data structure is a single instance of a quad-edge data structure.
#[derive(Clone, Debug)]
pub struct Heds<VData> {
    /// The vector of edges.
    pub edges: Slab<HEdge>,
    /// The vector of vertices.
    pub vertices: Slab<HVertex<VData>>,
    /// The vector of faces.
    pub faces: Slab<HFace>,
}

impl<VData> Heds<VData> {
    #[cfg(test)]
    fn assert_valid(&self) {
        // Each face should have only 3 edges.
        for (_, face) in self.faces.iter() {
            assert_eq!(face.all_edges(&self).len(), 3);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Location {
    OnVertex(HVertexRef),
    OnEdge(HEdgeRef),
    OnFace(HFaceRef),
}

impl<VData> Heds<VData> {
    pub fn new() -> Self {
        Self {
            edges: Slab::new(),
            vertices: Slab::new(),
            faces: Slab::new(),
        }
    }

    // TODO: This has not yet been proved to be stable. It may also loop
    // inifintely, particularly with constrianed triangulations.
    pub fn locate(&self, point: Point) -> Option<Location> {
        use rand::Rng;
        // There is always a zeroth edge if there are any.
        let mut e = HEdgeRef(0);
        let mut rng = rand::thread_rng();
        let mut current_iterations = 0;
        let edge = loop {
            current_iterations += 1;
            if current_iterations > 2000 {
                panic!("locating failed for: {}", point);
            }
            let edge = *self.edges.get(e.0).unwrap();
            let dest = *self.vertices.get(edge.vertex.0).unwrap();
            // TODO: this will crash often
            let pair = *self.edges.get(edge.pair.unwrap().0).unwrap();
            let origin = *self.vertices.get(pair.vertex.0).unwrap();
            // A random variable to determine if onext is tested first. If not
            // it is tested second.
            let onext_first: bool = rng.gen();
            // If the point is on the destination edge we're looking at, use
            // that.
            if point == dest.point {
                break Some(Location::OnVertex(edge.vertex));
            } else if point == origin.point {
                // If we lie on the eDest, return eSym(). This is a departure
                // from the paper which just returns e.
                break Some(Location::OnVertex(pair.vertex));
            } else if self.lies_right_strict(e, point) {
                // If the point lies to the right, take the pair so that it lies
                // to the left.
                e = edge.pair.unwrap();
            } else if onext_first {
                let next = edge.next;
                if !self.lies_right_strict(next, point) {
                    e = next;
                } else if !e.d_prev().lies_right_strict(point) {
                    e = e.d_prev();
                } else {
                    break Some(e);
                }
            } else if !e.d_prev().lies_right_strict(point) {
                e = e.d_prev();
            } else if !e.onext().lies_right_strict(point) {
                e = e.onext();
            } else {
                break Some(e);
            }
        };
        edge
    }
    pub fn lies_right_strict(&self, e: HEdgeRef, point: Point) -> bool {
        self.lies_right(e, point) == Lies::Yes
    }
    pub fn lies_right(&self, e: HEdgeRef, point: Point) -> Lies {
        use Lies::*;
        let edge = self.edges.get(e.0).unwrap();
        let pair = self.edges.get(edge.pair.unwrap().0).unwrap();
        let pa = self.vertices.get(edge.vertex.0).unwrap().point;
        let pb = self.vertices.get(pair.vertex.0).unwrap().point;
        match left_or_right(pa, pb, point) {
            Direction::Right => Yes,
            Direction::Straight => On,
            Direction::Left => No,
        }
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum Lies {
    Yes,
    No,
    On,
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum Direction {
    Left,
    Straight,
    Right,
}

/// Does p3 lie to the left or the right (or is collinear) of the line formed by
/// p1 and p2.
pub fn left_or_right(
    Point { x: x1, y: y1 }: Point,
    Point { x: x2, y: y2 }: Point,
    Point { x: x3, y: y3 }: Point,
) -> Direction {
    // TODO: check for overflow and underflow. If we use f32s we might be able
    // to rely on the hardware to do it for us, with conversions being cheaper
    // than branching. Actually we are probably ok with just checking for
    // infinity at the end.
    use Direction::*;
    let determinant = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    if !determinant.is_finite() {
        panic!("Non-finite determinant");
    } else if determinant > 0.0 {
        Left
    } else if determinant == 0.0 {
        Straight
    } else {
        Right
    }
}

impl<VData: Default> Heds<VData> {
    pub fn from_triangle_default(v1: Point, v2: Point, v3: Point) -> Self {
        let mut edges = Slab::new();
        edges.insert(HEdge {
            vertex: HVertexRef(0),
            pair: None,
            next: HEdgeRef(1),
            face: HFaceRef(0),
        });
        edges.insert(HEdge {
            vertex: HVertexRef(0),
            pair: None,
            next: HEdgeRef(1),
            face: HFaceRef(0),
        });
        edges.insert(HEdge {
            vertex: HVertexRef(0),
            pair: None,
            next: HEdgeRef(1),
            face: HFaceRef(0),
        });
        let mut vertices = Slab::with_capacity(3);
        vertices.insert(HVertex {
            point: v1,
            edge: HEdgeRef(0),
            data: Default::default(),
        });
        vertices.insert(HVertex {
            point: v2,
            edge: HEdgeRef(1),
            data: Default::default(),
        });
        vertices.insert(HVertex {
            point: v3,
            edge: HEdgeRef(2),
            data: Default::default(),
        });
        let mut faces = Slab::new();
        faces.insert(HFace { edge: HEdgeRef(0) });
        Self {
            edges,
            vertices,
            faces,
        }
    }

    /// The edge this returns should always have the added point at its origin.
    pub fn add_point_default(&mut self, mut point: Point) -> Option<HVertexRef> {
        point.snap();
        if let Some(location) = self.locate(point) {
            match location {
                Location::OnVertex(href) => Some(href),
                Location::OnEdge(href) => Some(self.add_to_edge_default(href, point)),
                Location::OnFace(href) => Some(self.add_to_face_default(href, point)),
            }
        } else {
            // Point was out of bounds (probably)
            None
        }
    }

    /// This function accounts for the point lying on an existing edge or point.
    fn add_to_face_default(&mut self, face: HFaceRef, point: Point) -> HVertexRef {
        let face_1_ref = face;
        let face = *self.faces.get(face.0).unwrap();
        let (old_edge_1, old_edge_2, old_edge_3) = face.tri_edges(self);
        let vertex_entry = self.vertices.vacant_entry();
        let vertex_ref = HVertexRef(vertex_entry.key());
        let face_2_ref = HFaceRef(self.faces.insert(Default::default()));
        let face_3_ref = HFaceRef(self.faces.insert(Default::default()));

        let edge_1_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_1b_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_2_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_2b_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_3_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_3b_ref = HEdgeRef(self.edges.insert(Default::default()));
        {
            let edge_1 = self.edges.get_mut(edge_1_ref.0).unwrap();
            *edge_1 = HEdge {
                vertex: vertex_ref,
                pair: Some(edge_1b_ref),
                face: face_1_ref,
                next: edge_3b_ref,
            };
        }
        {
            let edge_1b = self.edges.get_mut(edge_1b_ref.0).unwrap();
            *edge_1b = HEdge {
                vertex: HVertexRef(vertex_entry.key()),
                pair: Some(edge_1_ref),
                face: face_2_ref,
                next: old_edge_2,
            };
        }
        {
            let edge_2 = self.edges.get_mut(edge_2_ref.0).unwrap();
            *edge_2 = HEdge {
                vertex: HVertexRef(vertex_entry.key()),
                pair: Some(edge_2b_ref),
                face: face_2_ref,
                next: edge_2b_ref,
            };
        }
        {
            let edge_2b = self.edges.get_mut(edge_2b_ref.0).unwrap();
            *edge_2b = HEdge {
                vertex: HVertexRef(vertex_entry.key()),
                pair: Some(edge_2_ref),
                face: face_3_ref,
                next: old_edge_3,
            };
        }
        {
            let edge_3 = self.edges.get_mut(edge_3_ref.0).unwrap();
            *edge_3 = HEdge {
                vertex: HVertexRef(vertex_entry.key()),
                pair: Some(edge_3b_ref),
                face: face_3_ref,
                next: edge_2b_ref,
            };
        }
        {
            let edge_3b = self.edges.get_mut(edge_3b_ref.0).unwrap();
            *edge_3b = HEdge {
                vertex: HVertexRef(vertex_entry.key()),
                pair: Some(edge_3_ref),
                face: face_1_ref,
                next: old_edge_1,
            };
        }
        {
            let face_2 = self.faces.get_mut(face_2_ref.0).unwrap();
            *face_2 = HFace { edge: old_edge_2 };
        }
        {
            let face_3 = self.faces.get_mut(face_3_ref.0).unwrap();
            *face_3 = HFace { edge: old_edge_3 };
        }
        {
            let edge = self.edges.get_mut(old_edge_1.0).unwrap();
            edge.next = edge_1_ref;
        }
        {
            let edge = self.edges.get_mut(old_edge_2.0).unwrap();
            edge.next = edge_2_ref;
            edge.face = face_2_ref;
        }
        {
            let edge = self.edges.get_mut(old_edge_3.0).unwrap();
            edge.next = edge_3_ref;
            edge.face = face_3_ref;
        }
        vertex_entry.insert(HVertex {
            point,
            edge: edge_1_ref,
            data: Default::default(),
        });
        vertex_ref
    }

    /// Same as [`add_point_to_edge`] but does not check if the point is on one
    /// of the vertices of the edge.
    fn add_to_edge_default(&mut self, edge: HEdgeRef, point: Point) -> HVertexRef {
        let vertex_entry = self.vertices.vacant_entry();

        let vertex_ref = HVertexRef(vertex_entry.key());

        let edge_a_ref = edge;
        let edge_b_ref = self.edges.get(edge_a_ref.0).unwrap().pair.unwrap();

        let old_a_vertex = self.edges.get(edge_a_ref.0).unwrap().vertex;
        let old_b_vertex = self.edges.get(edge_b_ref.0).unwrap().vertex;
        let face_3_ref = HFaceRef(self.faces.insert(Default::default()));
        let edge_y_ref = self.edges.get(edge_b_ref.0).unwrap().next;

        let edge_ap_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_bp_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_c_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_cp_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_d_ref = HEdgeRef(self.edges.insert(Default::default()));
        let edge_dp_ref = HEdgeRef(self.edges.insert(Default::default()));

        {
            let edge_ap = self.edges.get_mut(edge_ap_ref.0).unwrap();
            edge_ap.vertex = old_b_vertex;
            edge_ap.pair = Some(edge_a_ref);
            edge_ap.face = face_3_ref;
            edge_ap.next = edge_y_ref;
        }

        let edge_w_ref = self.edges.get(edge_a_ref.0).unwrap().next;
        let edge_w_vertex = self.edges.get(edge_w_ref.0).unwrap().vertex;
        let edge_y_vertex = self.edges.get(edge_y_ref.0).unwrap().vertex;
        let edge_x_ref = self.edges.get(edge_w_ref.0).unwrap().next;
        let edge_z_ref = self.edges.get(edge_y_ref.0).unwrap().next;

        let face_1 = self.edges.get(edge_a_ref.0).unwrap().face;
        let face_2 = self.edges.get(edge_b_ref.0).unwrap().face;
        let face_4_ref = HFaceRef(self.faces.insert(Default::default()));

        {
            let edge_a = self.edges.get_mut(edge.0).unwrap();
            edge_a.vertex = vertex_ref;
            edge_a.pair = Some(edge_ap_ref);
            edge_a.next = edge_cp_ref;
        }
        // TODO: won't handle bounding edges
        {
            let edge_b = self.edges.get_mut(edge_b_ref.0).unwrap();
            edge_b.vertex = vertex_ref;
            edge_b.pair = Some(edge_bp_ref);
            edge_b.next = edge_dp_ref;
        }
        {
            let edge_bp = self.edges.get_mut(edge_bp_ref.0).unwrap();
            edge_bp.vertex = old_a_vertex;
            edge_bp.pair = Some(edge_b_ref);
            edge_bp.face = face_4_ref;
            edge_bp.next = edge_w_ref;
        }
        {
            let edge_c = self.edges.get_mut(edge_c_ref.0).unwrap();
            edge_c.vertex = vertex_ref;
            edge_c.pair = Some(edge_cp_ref);
            edge_c.face = face_4_ref;
            edge_c.next = edge_bp_ref;
        }
        {
            let edge_cp = self.edges.get_mut(edge_cp_ref.0).unwrap();
            edge_cp.vertex = edge_w_vertex;
            edge_cp.pair = Some(edge_c_ref);
            edge_cp.face = face_1;
            edge_cp.next = edge_x_ref;
        }
        {
            let edge_d = self.edges.get_mut(edge_d_ref.0).unwrap();
            edge_d.vertex = vertex_ref;
            edge_d.pair = Some(edge_dp_ref);
            edge_d.face = face_3_ref;
            edge_d.next = edge_ap_ref;
        }
        {
            let edge_dp = self.edges.get_mut(edge_dp_ref.0).unwrap();
            edge_dp.vertex = edge_y_vertex;
            edge_dp.pair = Some(edge_d_ref);
            edge_dp.face = face_2;
            edge_dp.next = edge_z_ref;
        }
        {
            let edge_w = self.edges.get_mut(edge_w_ref.0).unwrap();
            edge_w.face = face_4_ref;
            edge_w.next = edge_c_ref;
        }
        {
            let edge_x = self.edges.get_mut(edge_x_ref.0).unwrap();
            edge_x.face = face_1;
            edge_x.next = edge_a_ref;
        }
        {
            let edge_y = self.edges.get_mut(edge_y_ref.0).unwrap();
            edge_y.face = face_3_ref;
            edge_y.next = edge_d_ref;
        }
        {
            let edge_z = self.edges.get_mut(edge_z_ref.0).unwrap();
            edge_z.face = face_2;
            edge_z.next = edge_b_ref;
        }
        {
            let face_3 = self.faces.get_mut(face_3_ref.0).unwrap();
            face_3.edge = edge_y_ref;
        }
        {
            let face_4 = self.faces.get_mut(face_4_ref.0).unwrap();
            face_4.edge = edge_w_ref;
        }

        vertex_entry.insert(HVertex {
            edge: edge_ap_ref,
            point,
            data: Default::default(),
        });

        vertex_ref
    }
}

impl<VData> Default for Heds<VData> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Default)]
pub struct HEdge {
    /// Vertex at the end of the half edge.
    pub vertex: HVertexRef,
    /// Oppositely oriented adjacent half edge. None if it is a boundary.
    pub pair: Option<HEdgeRef>,
    /// Face the half edge borders.
    pub face: HFaceRef,
    /// Next half edge around the face (CCW).
    pub next: HEdgeRef,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Default)]
pub struct HVertex<VData> {
    pub point: Point,
    /// A reference to one of the half edges emanating from the vertex.
    pub edge: HEdgeRef,
    /// The data stored at this vertex.
    pub data: VData,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Default)]
pub struct HFace {
    /// A reference to one of the half edges bording the face.
    pub edge: HEdgeRef,
}

impl HFace {
    pub fn new(edge: HEdgeRef) -> Self {
        Self { edge }
    }
    pub fn tri_edges<T>(&self, heds: &Heds<T>) -> (HEdgeRef, HEdgeRef, HEdgeRef) {
        let edge_1 = self.edge;
        let edge_2 = heds.edges.get(edge_1.0).unwrap().next;
        let edge_3 = heds.edges.get(edge_2.0).unwrap().next;
        (edge_1, edge_2, edge_3)
    }
    pub fn all_edges<T>(&self, heds: &Heds<T>) -> Vec<HEdgeRef> {
        let mut edges = Vec::new();
        let first_edge = self.edge;
        edges.push(first_edge);
        let mut current_edge = first_edge;
        loop {
            let next_edge = heds.edges.get(current_edge.0).unwrap().next;
            if next_edge == first_edge {
                break;
            }
            edges.push(next_edge);
            current_edge = next_edge;
            if edges.len() > 3 {
                panic!("too many edges to face, should only have 3");
            }
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn creation() {
        Heds::<()>::new();
    }

    #[test]
    fn create_tri() {
        let v1 = Point::new(0.0, 0.0);
        let v2 = Point::new(5.0, 0.0);
        let v3 = Point::new(5.0, 5.0);
        let heds = Heds::<f64>::from_triangle_default(v1, v2, v3);
        assert_eq!(heds.vertices.len(), 3);
        assert_eq!(heds.edges.len(), 3);
        assert_eq!(heds.faces.len(), 1);
    }

    #[test]
    fn tri_insertion() {
        let v1 = Point::new(0.0, 0.0);
        let v2 = Point::new(5.0, 0.0);
        let v3 = Point::new(5.0, 5.0);
        let mut heds = Heds::<f64>::from_triangle_default(v1, v2, v3);
        heds.add_to_face_default(HFaceRef(0), Point::new(5.0 * 2.0 / 3.0, 5.0 * 1.0 / 3.0));
        assert_eq!(heds.vertices.len(), 4);
        assert_eq!(heds.edges.len(), 9);
        assert_eq!(heds.faces.len(), 3);
        heds.assert_valid();
    }
}
