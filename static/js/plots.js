

$("#btnregsubmit").click(function(){
	debugger;
	var uname=document.getElementById('uname').value;
	var name=document.getElementById('name').value;
	var pswd=document.getElementById('pswd').value;
	var email=document.getElementById('email').value;
	var phone=document.getElementById('phone').value;
	var addr=document.getElementById('addr').value;
	var utype=document.getElementById('utype').value;
	

        var flag = true;

         var intRegexunamer = /^[A-Za-z0-9 ]+$/;
         if (!intRegexunamer.test(uname)) {
             alert('Please enter a valid username.');
             flag = false;
             return flag;
         }
         else {
             flag = true;
         }

         var intRegexnamer = /^[A-Za-z ]+$/;
         if (!intRegexnamer.test(name)) {
             alert('Please enter a valid name.');
             flag = false;
             return flag;
         }
         else {
             flag = true;
         }

         var pswdPattern = /^[A-Za-z0-9]{5,8}$/;
         if (!pswdPattern.test(pswd)) {
             alert('Please enter 8 characters password.');
             flag = false;
             return flag;
         }
         else {
             flag = true;
         }
         

         var emailReg = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
         if (!emailReg.test(email) || email == '') {
             alert('Please enter a valid email id.');
             flag = false;
             return flag;
         }
         else {
             flag = true;
         }

         var intRegex = /^(7|8|9)[0-9]{9}$/;
         if (!intRegex.test(phone)) {
             alert('Please enter a valid phone number.');
             flag = false;
             return flag;
         }
         else {
             flag = true;
         }

         if (addr=="") {
            alert('Please enter a valid address.');
            flag = false;
            return flag;
        }
        else {
            flag = true;
        }

        if (utype=="Select User Type") {
           alert('Please select user type');
           flag = false;
           return flag;
       }
       else {
           flag = true;
       }




       
	
	
	//var gender="";
	//if(document.getElementById('gen').checked==true)
	//	gender="Male";
	//if(document.getElementById('gen1').checked==true)
	//	gender="Female";
	
	/* window.location='regdata?uname='+uname+'&name='+name+'&pswd='+pswd+'&email='+email+'&phone='+phone+'&addr='+addr;*/
	
	$.ajax({
            type: 'GET',
            url: '/regdata',
			
        contentType: 'application/json;charset=UTF-8',
            data: {
            'uname': uname,
            'name': name,
            'email': email,
            'phone': phone,
            'pswd': pswd,
            'addr': addr,
            'utype': utype
			

        },
            
        dataType:"json",
            success: function(data) {
				alert('User Account Created Successfully');
               window.location='login';
            },
        });
	
});

$("#btnplancluster").click(function(){
	debugger;
	var ven=document.getElementById('venclus').value;
	
	
	window.location='/gencluster?ven='+ven;
});


$("#btnrideforecast").click(function(){
	debugger;
	var ven=document.getElementById('venfor').value;
	
	
	window.location='/genforecast?ven='+ven;
});

$("#btnpredict").click(function(){
	debugger;
	var loc=document.getElementById('loc').value;
	
	
	window.location='/locdata?loc='+loc;
	
	//var gender="";
	//if(document.getElementById('gen').checked==true)
	//	gender="Male";
	//if(document.getElementById('gen1').checked==true)
	//	gender="Female";
	
	/* window.location='regdata?uname='+uname+'&name='+name+'&pswd='+pswd+'&email='+email+'&phone='+phone+'&addr='+addr;*/
	
/*	$.ajax({
            type: 'GET',
            url: '/locdata',
			
        contentType: 'application/json;charset=UTF-8',
            data: {
            'loc': loc	

        },
            
        dataType:"json",
            success: function(data) {
				alert('Data saved Successfully');
				acheck();
              // window.location='register';
            },
        });*/
	
});



$("#btnforgotpassword").click(function(){
	debugger;
	var email=document.getElementById('email').value;
	
	
	$.ajax({
            type: 'GET',
            url: '/fpassword',
			
        contentType: 'application/json;charset=UTF-8',
            data: {
            'email': email
                },
            
        dataType:"json",
            success: function(data) {
					alert('Mail Sent Successfully');
				   window.location='forgotpassword';
            },
			 error: function(data) {
               
            }
        });
	
});



$("#btndata").click(function(){
	debugger;
	var dataid=document.getElementById('dataid').value;

    
	$.ajax({
        type: 'GET',
        url: '/validate',
        
    contentType: 'application/json;charset=UTF-8',
        data: {
        'dataid': dataid
            },
        
    dataType:"json",
        success: function(data) {
                document.getElementById('res').innerHTML=data;
        },
         error: function(data) {
           
        }
    });
});

$("#btnlogsubmit").click(function(){
	debugger;
	var email=document.getElementById('email').value;
	var pswd=document.getElementById('pswd').value;
	
	if(email=="admin@gmail.com" && pswd=="admin")
    {
        window.location="adminhome"
    }
    else{

        var emailReg = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
        if (!emailReg.test(email) || email == '') {
            alert('Please enter a valid email id.');
            flag = false;
            return flag;
        }
        else {
            flag = true;
        }

        
        var pswdPattern = /^[A-Za-z0-9]{5,8}$/;
        if (!pswdPattern.test(pswd)) {
            alert('Please enter 8 characters password.');
            flag = false;
            return flag;
        }
        else {
            flag = true;
        }
        

       
	$.ajax({
            type: 'GET',
            url: '/logdata',
			
        contentType: 'application/json;charset=UTF-8',
            data: {
            'email': email,
            'pswd': pswd
			

        },
            
        dataType:"json",
            success: function(data) {
				if(data=="Failure")
				{
					alert("Credentials not found");
					window.location='login';
				}
				if(data=="Invigilator")
				{
					alert('Logged in Successfully');
				   window.location='dashboard';
				}
				if(data=="Student")
				{
					alert('Logged in Successfully');
				   window.location='student';
				}
            },
			 error: function(data) {
               
            }
        });
    }
	
});

function acheck()
{
	debugger;
}



$("#dataload_btnsubmit").click(function(){
	debugger;
   var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/uploadajax',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
				alert('Data stored successfully');
            },
        });
});

$("#dataload_btnclear").click(function(){
   var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/cleardataset',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
				alert('Dataset has been cleared');
            },
        });
});
