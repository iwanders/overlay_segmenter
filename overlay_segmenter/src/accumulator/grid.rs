use flash_powder::prelude::*;
use serde::{Deserialize, Serialize};
#[derive(Copy, Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct Position {
    pub x: isize,
    pub y: isize,
}
impl Position {
    pub fn new(x: isize, y: isize) -> Self {
        Self { x, y }
    }

    pub fn origin() -> Self {
        Self { x: 0, y: 0 }
    }
    pub fn min(&self, other: Position) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }
    pub fn max(&self, other: Position) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }
}

impl From<Position> for (isize, isize) {
    fn from(s: Position) -> Self {
        (s.x, s.y)
    }
}

impl From<(isize, isize)> for Position {
    fn from(s: (isize, isize)) -> Self {
        Position { x: s.0, y: s.1 }
    }
}

impl std::ops::Add for Position {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Position {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl std::ops::Sub for Position {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Position {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct Rect {
    pub w: usize,
    pub h: usize,
}

impl From<(usize, usize)> for Rect {
    fn from(s: (usize, usize)) -> Self {
        Rect { w: s.0, h: s.1 }
    }
}

impl std::ops::Add<Rect> for Position {
    type Output = Position;

    fn add(self, rhs: Rect) -> Self::Output {
        Position {
            x: self.x + rhs.w as isize,
            y: self.y + rhs.h as isize,
        }
    }
}
impl std::ops::Add<Position> for Rect {
    type Output = Position;

    fn add(self, rhs: Position) -> Self::Output {
        Position {
            x: rhs.x + self.w as isize,
            y: rhs.y + self.h as isize,
        }
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct GridId(usize);

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct GridWindow {
    size: Rect,
    position: Position,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GridOverlay {
    windows: Vec<GridWindow>,
}
impl GridOverlay {
    pub fn new() -> Self {
        Self { windows: vec![] }
    }
    pub fn ids(&self) -> impl Iterator<Item = GridId> {
        (0..self.windows.len()).into_iter().map(|a| GridId(a))
    }

    /// Add a grid.
    pub fn add_grid_raw(&mut self, size: (usize, usize), position: Position) -> GridId {
        let id = GridId(self.windows.len());
        self.windows.push(GridWindow {
            size: Rect {
                w: size.0,
                h: size.1,
            },
            position: Position::new(position.x, position.y),
        });
        id
    }
    /// Add a tensor grid
    pub fn add_tensor<T: TensorProperties>(&mut self, t: &T, position: Position) -> GridId {
        let id = GridId(self.windows.len());
        self.windows.push(GridWindow {
            size: Rect {
                w: t.isize(-1) as _,
                h: t.isize(-2) as _,
            },
            position: Position::new(position.x, position.y),
        });
        id
    }

    pub fn extent(&self) -> (Position, Position) {
        if self.windows.is_empty() {
            return (Position::origin(), Position::origin());
        }
        let mut min = self.windows.first().unwrap().position;
        let mut max = self.windows.first().unwrap().position;
        for w in self.windows.iter() {
            min = min.min(w.position);
            max = max.max(w.position + w.size)
        }

        (min, max)
    }

    pub fn overlap(&self) -> (Position, Position) {
        if self.windows.is_empty() {
            return (Position::origin(), Position::origin());
        }
        let mut min = self.windows.first().unwrap().position;
        let mut max = self.windows.first().unwrap().position + self.windows.first().unwrap().size;
        for w in self.windows.iter() {
            min = min.max(w.position);
            max = max.min(w.position + w.size)
        }

        (min, max)
    }

    pub fn full_size(&self) -> (usize, usize) {
        let (min, max) = self.extent();
        let diff = max - min;
        (diff.x as usize, diff.y as usize)
    }

    pub fn full_position(&self) -> (isize, isize) {
        let (min, max) = self.extent();
        min.into()
    }

    /// Returns the position of this grid in the full size coordinates.
    pub fn full_grid_position(&self, grid: GridId) -> (usize, usize) {
        let (min, _max) = self.extent();
        for (i, v) in self.windows.iter().enumerate() {
            if i == grid.0 {
                let pos = v.position;
                let this_pos = pos - min;
                return (this_pos.x as usize, this_pos.y as usize);
            }
        }
        unreachable!("grid id {grid:?} was passed, which doesn't originate from this GridOverlay");
    }
    /// Returns the position of this grid in the full size coordinates.
    pub fn full_grid_irange(
        &self,
        grid: GridId,
    ) -> (std::ops::Range<isize>, std::ops::Range<isize>) {
        let (min, _max) = self.extent();
        for (i, v) in self.windows.iter().enumerate() {
            if i == grid.0 {
                let pos = v.position;
                let this_pos = pos - min;
                return (
                    this_pos.x..(this_pos.x + v.size.w as isize),
                    this_pos.y..(this_pos.y + v.size.h as isize),
                );
            }
        }
        unreachable!("grid id {grid:?} was passed, which doesn't originate from this GridOverlay");
    }

    pub fn grid_position(&self, grid: GridId) -> Position {
        for (i, v) in self.windows.iter().enumerate() {
            if i == grid.0 {
                return v.position;
            }
        }
        unreachable!("grid id {grid:?} was passed, which doesn't originate from this GridOverlay");
    }
    /// Returns the area of this grid that is part of the overlap.
    pub fn overlap_irange(&self, grid: GridId) -> (std::ops::Range<isize>, std::ops::Range<isize>) {
        let (min, max) = self.overlap();

        for (i, v) in self.windows.iter().enumerate() {
            if i == grid.0 {
                let this_pos = v.position;
                return (
                    min.x - this_pos.x..max.x - this_pos.x,
                    min.y - this_pos.y..max.y - this_pos.y,
                );
            }
        }
        unreachable!("grid id {grid:?} was passed, which doesn't originate from this GridOverlay");
    }

    /// Calculates the necessary growth for the other pos & size to fit on a new canvas together with current data.
    ///
    /// Returns new grid position and size if growth is necessary, empty if the current grid is large enough to hold the
    /// new data.
    pub fn determine_growth(
        &self,
        other_pos: Position,
        other_size: Rect,
    ) -> Option<(Position, Rect)> {
        let mut combined_grid = GridOverlay::new();
        combined_grid.add_grid_raw((other_size.w, other_size.h), other_pos);
        combined_grid.add_grid_raw(self.full_size(), combined_grid.full_position().into());

        // If extents are equal, we don't need to do anything because it already fits, no growth happened.
        if combined_grid.extent() == self.extent() {
            return None;
        }

        // The extent was not the same, so just return the combined grid and position.
        Some((
            combined_grid.full_position().into(),
            combined_grid.full_size().into(),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_overlay() {
        // First test, trivial, to the right, completely outside of.
        let base = (3, 5);
        let overlay = (1, 3);
        let position = (8, 7);
        let mut o = GridOverlay::new();
        let b_id = o.add_grid_raw(base, Position::origin());
        let o_id = o.add_grid_raw(overlay, position.into());

        let (lowest, highest) = o.extent();
        assert_eq!(lowest.x, 0); // start of base.
        assert_eq!(lowest.y, 0);
        assert_eq!(highest.x, 1 + 8);
        assert_eq!(highest.y, 3 + 7);

        let (w, h) = o.full_size();
        assert_eq!(w, 1 + 8);
        assert_eq!(h, 3 + 7);

        let b_in_full = o.full_grid_position(b_id);
        assert_eq!(b_in_full.0, 0);
        assert_eq!(b_in_full.1, 0);
        let o_in_full = o.full_grid_position(o_id);
        assert_eq!(o_in_full.0, 8);
        assert_eq!(o_in_full.1, 7);

        // Now, lets place the overlay in the lower quadrant relative to base.
        let base = (3, 5);
        let overlay = (1, 3);
        let position = (-8, -7);
        let mut o = GridOverlay::new();
        let b_id = o.add_grid_raw(base, (0, 0).into());
        let o_id = o.add_grid_raw(overlay, position.into());

        let (lowest, highest) = o.extent();
        assert_eq!(lowest.x, -8); // Start of overlay
        assert_eq!(lowest.y, -7); //
        assert_eq!(highest.x, 3);
        assert_eq!(highest.y, 5);

        let (w, h) = o.full_size();
        assert_eq!(w, (-8isize - 3isize).abs() as usize);
        assert_eq!(h, (-7isize - 5isize).abs() as usize);

        let b_in_full = o.full_grid_position(b_id);
        assert_eq!(b_in_full.0, 8);
        assert_eq!(b_in_full.1, 7);
        let o_in_full = o.full_grid_position(o_id);
        assert_eq!(o_in_full.0, 0);
        assert_eq!(o_in_full.1, 0);

        // Now, lets put the base not at the origin.
        let base = (3, 5);
        let base_pos = (-3, -5);
        let overlay = (1, 3);
        let position = (-8, -7);
        let mut o = GridOverlay::new();
        let b_id = o.add_grid_raw(base, base_pos.into());
        let o_id = o.add_grid_raw(overlay, position.into());

        let (lowest, highest) = o.extent();
        assert_eq!(lowest.x, -8); // Start of overlay
        assert_eq!(lowest.y, -7); //
        assert_eq!(highest.x, 0);
        assert_eq!(highest.y, 0);

        let (w, h) = o.full_size();
        assert_eq!(w, (-8isize - 0isize).abs() as usize);
        assert_eq!(h, (-7isize - 0isize).abs() as usize);

        // Full starts at -8,-7, base is at -3, -5
        let b_in_full = o.full_grid_position(b_id);
        assert_eq!(b_in_full.0, (-8isize - -3isize).abs() as usize);
        assert_eq!(b_in_full.1, (-7isize - -5isize).abs() as usize);
        let o_in_full = o.full_grid_position(o_id);
        assert_eq!(o_in_full.0, 0);
        assert_eq!(o_in_full.1, 0);

        // Those were all disjoint, so now lets make something that's not disjoint.
        let base = (4, 5);
        let base_pos = (-1, -1);
        //  0  1  2  3 ;  0  1  2  3  4
        // -1, 0, 1  2;  -1, 0, 1, 2, 3
        let overlay = (3, 4);
        let position = (1, 2);
        // 0 1 2 ; 0 1 2 3
        // 1 2 3 ; 2 3 4 5
        let mut o = GridOverlay::new();
        let b_id = o.add_grid_raw(base, base_pos.into());
        let o_id = o.add_grid_raw(overlay, position.into());
        let (min, max) = o.overlap();
        assert_eq!(min.x, 1);
        assert_eq!(min.y, 2);

        // One beyond the overlap, of course.
        assert_eq!(max.x, 3);
        assert_eq!(max.y, 4);

        let (b_overlap_x, b_overlap_y) = o.overlap_irange(b_id);
        assert_eq!(b_overlap_x, 2..4);
        assert_eq!(b_overlap_y, 3..5);
        let (o_overlap_x, o_overlap_y) = o.overlap_irange(o_id);
        assert_eq!(o_overlap_x, 0..2);
        assert_eq!(o_overlap_y, 0..2);

        // Next, test the determine_growth function.
        let mut o = GridOverlay::new();
        let b_id = o.add_grid_raw((10, 10), Position::origin());
        let inside = o.determine_growth(Position { x: 0, y: 0 }, Rect { w: 5, h: 5 });
        assert_eq!(inside, None);

        // Something to the left.
        let outside = o.determine_growth(Position { x: -5, y: 0 }, Rect { w: 3, h: 5 });
        assert!(outside.is_some());
        let (p, s) = outside.unwrap();
        assert_eq!(p, Position { x: -5, y: 0 });
        assert_eq!(s, Rect { w: 10, h: 10 });
    }
}
