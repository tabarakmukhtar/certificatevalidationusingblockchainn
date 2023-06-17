-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 12, 2023 at 03:04 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `flaskedubcdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `certdata`
--

CREATE TABLE `certdata` (
  `id` int(11) NOT NULL,
  `Course` varchar(50) DEFAULT NULL,
  `Domain` varchar(50) DEFAULT NULL,
  `College` varchar(50) DEFAULT NULL,
  `Semester` varchar(50) DEFAULT NULL,
  `stname` varchar(50) DEFAULT NULL,
  `USN` varchar(50) DEFAULT NULL,
  `SubjectCode` varchar(50) DEFAULT NULL,
  `SName` varchar(50) DEFAULT NULL,
  `AssignedCredits` varchar(50) DEFAULT NULL,
  `ObtainedCredits` varchar(50) DEFAULT NULL,
  `Email` varchar(50) DEFAULT NULL,
  `Stat` varchar(50) NOT NULL,
  `Template` varchar(500) NOT NULL,
  `uploadedby` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `certdata`
--

INSERT INTO `certdata` (`id`, `Course`, `Domain`, `College`, `Semester`, `stname`, `USN`, `SubjectCode`, `SName`, `AssignedCredits`, `ObtainedCredits`, `Email`, `Stat`, `Template`, `uploadedby`) VALUES
(607, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18MAT31', 'Transform calculus', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(608, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18CS32', 'DataStructures using C', '4', '3', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(609, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18CS33', 'ADE', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(610, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18CS34', 'CO', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(611, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18CS35', 'SE', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(612, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18CS36', 'DMS', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(613, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18CSL37', 'ADE LAB', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(614, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18CSL38', 'DS LAB', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(615, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Athiq', '4MH19IS012', '18KAK38', 'ELECTIVE', '3', '2', 'athiqap8453@gmail.com', 'Mail Sent', 'vtucert.jpg', 'inv@gmail.com'),
(616, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO36', '18CS32', 'DataStructures using C', '4', '3', '23vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(617, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO37', '18CS33', 'ADE', '3', '3', '24vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(618, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO38', '18CS34', 'CO', '3', '3', '25vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(619, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO39', '18CS35', 'SE', '3', '3', '26vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(620, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO40', '18CS36', 'DMS', '3', '3', '27vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(621, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO41', '18CSL37', 'ADE LAB', '3', '3', '28vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(622, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO42', '18CSL38', 'DS LAB', '3', '3', '29vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(623, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO43', '18MAT31', 'Transform calculus', '3', '3', '30vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(624, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Bindu', '4MH19ISO44', '18KAK38', 'ELECTIVE', '3', '3', '31vagdeviv@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(625, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18CS32', 'DataStructures using C', '4', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(626, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18CS33', 'ADE', '3', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(627, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18CS34', 'CO', '3', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(628, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18CS35', 'SE', '3', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(629, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18CS36', 'DMS', '3', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(630, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18CSL37', 'ADE LAB', '3', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(631, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18CSL38', 'DS LAB', '3', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com'),
(632, 'B.E.', 'Information Science', 'MITM', 'III Semester', 'Adhira', '4MO19ISO20', '18MAT31', 'ELECTIVE', '3', '3', 'darshanrameshcr@gmail.com', 'Pending', 'vtucert.jpg', 'inv@gmail.com');

-- --------------------------------------------------------

--
-- Table structure for table `mdata`
--

CREATE TABLE `mdata` (
  `id` int(11) NOT NULL,
  `certval` int(11) DEFAULT NULL,
  `image` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `mdata`
--

INSERT INTO `mdata` (`id`, `certval`, `image`) VALUES
(3, 2088697, 'Athiq.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `userdata`
--

CREATE TABLE `userdata` (
  `Uid` varchar(50) NOT NULL,
  `Uname` varchar(80) NOT NULL,
  `Name` varchar(50) NOT NULL,
  `Pswd` varchar(50) NOT NULL,
  `Email` varchar(50) NOT NULL,
  `Phone` varchar(50) NOT NULL,
  `Addr` varchar(500) NOT NULL,
  `utype` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `userdata`
--

INSERT INTO `userdata` (`Uid`, `Uname`, `Name`, `Pswd`, `Email`, `Phone`, `Addr`, `utype`) VALUES
('User77109', 'Inv', 'Inv', 'qazwsx', 'inv@gmail.com', '9036453696', 'Mysore', 'Invigilator'),
('User87905', 'Tarun', 'N', 'qazwsx', 'tarun@gmail.com', '9844921346', 'sdfghjk', 'Invigilator'),
('User13933', 'tabarak', 'Tabarak Mukhtar', 'Tabarak7', 'tabarakmukhtar159@gmail.com', '9099990909', 'Near amaya Bakery Giridaeshini Layout', 'Student'),
('User60146', 'athiq', 'Athiq Pasha', 'Athiqpas', 'athiqap8453@gmail.com', '9740679137', 'bannimantap mysore', 'Student'),
('User4828', 'BINDU', 'Bindushree TR', 'Bindushr', 'bindu2002@gmail.com', '7795992667', 'kolegala', 'Student'),
('User52127', 'Athiq', 'Athiq', 'qazwsx', 'athiq@gmail.com', '9036453696', 'Demo', 'Student');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `certdata`
--
ALTER TABLE `certdata`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `mdata`
--
ALTER TABLE `mdata`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `certdata`
--
ALTER TABLE `certdata`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=633;

--
-- AUTO_INCREMENT for table `mdata`
--
ALTER TABLE `mdata`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
