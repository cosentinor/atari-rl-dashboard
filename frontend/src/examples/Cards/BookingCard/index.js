/**
=========================================================
* Material Dashboard 3 PRO React - v2.4.0
=========================================================

* Product Page: https://www.creative-tim.com/product/material-dashboard-pro-react
* Copyright 2024 Creative Tim (https://www.creative-tim.com)

Coded by www.creative-tim.com

 =========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

// prop-types is a library for typechecking of props
import PropTypes from "prop-types";

// @mui material components
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";
import Icon from "@mui/material/Icon";

// Material Dashboard 3 PRO React components
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";

function BookingCard({ 
  image = "",
  title = "",
  description = "",
  price = "",
  location = "",
  action = null 
}) {
  return (
    <Card sx={{
      "&:hover .card-header": {
        transform: action && "translate3d(0, -50px, 0)",
      },
    }}>
      <MDBox position="relative" borderRadius="md" mx={1} mt={1} className="card-header"
        sx={{ transition: "transform 300ms cubic-bezier(0.34, 1.61, 0.7, 1)" }}>
        <MDBox
          component="img"
          src={image}
          alt={title}
          borderRadius="md"
          shadow="md"
          width="100%"
          height="100%"
          position="relative"
          zIndex={1}
        />
        <MDBox
          borderRadius="md"
          shadow="md"
          width="100%"
          height="100%"
          position="absolute"
          left={0}
          top="0"
          sx={{
            backgroundImage: `url(${image})`,
            transform: "scale(0.94)",
            filter: "blur(12px)",
            backgroundSize: "cover",
          }}
        />
      </MDBox>
      <MDBox pt={3} px={3}>
        <MDBox
          display="flex"
          justifyContent="center"
          alignItems="center"
          mt={action ? -8 : -4.25}
        >
          {action}
        </MDBox>
        <MDTypography variant="h5" fontWeight="bold" sx={{ mt: 4 }}>
          {title}
        </MDTypography>
        <MDTypography
          variant="body2"
          fontWeight="regular"
          color="text"
          sx={{ mt: 1.5, mb: 1, fontSize: "14px" }}
        >
          {description}
        </MDTypography>
      </MDBox>
      <Divider />
      <MDBox
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        px={2}
        pb={1.5}
        lineHeight={1}
      >
        <MDBox color="text" display="flex" alignItems="center">
          <Icon color="inherit" sx={{ m: 0.5 }}>
            place
          </Icon>
          <MDTypography variant="button" fontWeight="light" color="text">
            {location}
          </MDTypography>
        </MDBox>
        <MDTypography variant="body2" fontWeight="medium">
          {price}
        </MDTypography>
      </MDBox>
    </Card>
  );
}

// Typechecking props for the BookingCard
BookingCard.propTypes = {
  image: PropTypes.string.isRequired,
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  price: PropTypes.string.isRequired,
  location: PropTypes.node.isRequired,
  action: PropTypes.oneOfType([PropTypes.bool, PropTypes.node]),
};

export default BookingCard;
