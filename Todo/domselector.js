var button = document.getElementById("enter");
var input = document.getElementById("userinput");
var ul = document.querySelector("ul");
var items = document.querySelectorAll("li");
var clearbutton = document.getElementById("clearbutton")

function inputLength() {
  return input.value.length;
}

// new list items
function createListElement() {

  var li = document.createElement("li");
  li.appendChild(document.createTextNode(input.value));

  //creates buttons
  var btn1 = document.createElement("button");
  var btn2 = document.createElement("button");
  btn1.innerHTML = "Done";
  btn2.innerHTML = "Delete";
  li.appendChild(btn1);
  li.appendChild(btn2);
  // done button work
  btn1.addEventListener("click", function() {
    li.style.opacity = 0.5;
  })

  // removes element
  btn2.addEventListener("click", function () {
    li.parentNode.removeChild(li);
  });


  ul.appendChild(li);
  input.value = "";

  
}
//clear button 
{
var clearbtn = document.createElement("button")
clearbtn.innerHTML="Clear";
clearbutton.appendChild(clearbtn)

clearbtn.addEventListener("click", function(){
  while(ul.firstChild){
    ul.removeChild(ul.firstChild);
  }
})
}


function addListAfterClick() {
  if (inputLength() > 0) {
    createListElement();
  }
  
}

function addListAfterKeypress(event) {
  if (inputLength() > 0 && event.keyCode === 13) {
    createListElement();
  }
}

button.addEventListener("click", addListAfterClick);
input.addEventListener("keypress", addListAfterKeypress);


