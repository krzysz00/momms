//Public Modules
mod matrix;
mod general_stride;
mod row_panel;
mod column_panel;
mod hierarch;
mod pack_pair;
mod subcomputation;
mod metadata_only;

pub use self::matrix::{Scalar,Mat,ResizableBuffer,RoCM,PanelTranspose};
pub use self::general_stride::{Matrix};
pub use self::row_panel::{RowPanelMatrix};
pub use self::column_panel::{ColumnPanelMatrix};
pub use self::hierarch::{Hierarch,HierarchyNode};
pub use self::pack_pair::{PackPair};
pub use self::subcomputation::{Subcomputation,TransposingSubcomputation};
pub use self::metadata_only::{MetadataOnlyMatrix};
//Private Modules
mod view;
