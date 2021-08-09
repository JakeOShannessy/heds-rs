use slab::Slab;
mod point;
pub use point::*;

/// Offset into the vertex array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HVertexRef(usize);

/// Offset into the edge array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HEdgeRef(usize);

/// Offset into the face array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    pub fn new() -> Self {
        Self {
            edges: Slab::new(),
            vertices: Slab::new(),
            faces: Slab::new(),
        }
    }
}

impl<VData> Default for Heds<VData> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HEdge {
    /// Vertex at the end of the half edge.
    pub vertex: HVertexRef,
    /// Oppositely oriented adjacent half edge.
    pub pair: HVertexRef,
    /// Face the half edge borders.
    pub face: HFaceRef,
    /// Next half edge around the face.
    pub next: HEdgeRef,
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HVertex<VData> {
    pub point: Point,
    /// A reference to one of the half edges emanating from the vertex.
    pub edge: HEdgeRef,
    /// The data stored at this vertex.
    pub data: VData,
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HFace {
    /// A reference to one of the half edges bording the face.
    pub edge: HEdgeRef,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn creation() {
        Heds::<()>::new();
    }
}
