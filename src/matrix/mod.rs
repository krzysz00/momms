//Public Modules
mod matrix;
mod general_stride;
mod row_panel;
mod column_panel;
mod hierarch;
mod void;

pub use self::matrix::{Scalar,Mat,ResizableBuffer};
pub use self::general_stride::{Matrix};
pub use self::row_panel::{RowPanelMatrix};
pub use self::column_panel::{ColumnPanelMatrix};
pub use self::hierarch::{Hierarch};
pub use self::void::{VoidMat};

//Private Modules
mod view;
